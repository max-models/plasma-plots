"""Numerical diagnostics returning values and labeled arrays, without rendering.

Fits (growth and damping rates, convergence orders, dispersion branches), norms and errors,
volume integrals and field energies on mapped domains, vector calculus on the logical grid,
velocity moments and orbit diagnostics. Most functions are also available as accessor methods,
e.g. ``array.plasma.analysis.error(...)``.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .arrays import angle_period, logical_dims, validate_array


def _label(data):
    return data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""


@dataclass(frozen=True)
class GrowthFit:
    """Configuration for an exponential growth-rate fit.

    Passed to :func:`growth_rate` and :func:`damping_rate`.

    Attributes
    ----------
    window : (float or None, float or None)
        The time interval ``(start, end)`` of the samples that are fitted; ``None`` means the
        first or last time. The two ends may be given in either order. Default: every sample.
    amplitude_from_quadratic : bool
        Whether the series is quadratic in an amplitude (e.g. an energy). The fit then uses the
        square root of the samples, so that ``rate`` is the growth rate of the amplitude, half that
        of the series itself. Default: ``False``.
    """

    window: tuple[float | None, float | None] = (None, None)
    amplitude_from_quadratic: bool = False


@dataclass(frozen=True)
class FitResult:
    """An exponential fit ``exp(rate t + intercept)``, from :func:`growth_rate` or :func:`damping_rate`.

    Attributes
    ----------
    rate : float
        The fitted rate: positive for growth, negative for damping. With
        ``GrowthFit.amplitude_from_quadratic`` it is the rate of the amplitude.
    intercept : float
        The fitted intercept of the logarithm (of the amplitude, with
        ``GrowthFit.amplitude_from_quadratic``).
    time : numpy.ndarray
        The times of the samples that were used in the fit.
    fitted : numpy.ndarray
        The fitted curve at ``time``, in the units of the fitted series (squared again with
        ``GrowthFit.amplitude_from_quadratic``), ready to plot over the data.
    """

    rate: float
    intercept: float
    time: np.ndarray
    fitted: np.ndarray


@dataclass(frozen=True)
class ConvergenceFit:
    """A power law ``error = constant * size**order``, from :func:`convergence_order`.

    Attributes
    ----------
    order : float
        The fitted exponent: negative when the error shrinks as the size grows (e.g. points per
        cell), positive when it shrinks with the size (e.g. ``dt``).
    constant : float
        The fitted prefactor.
    sizes : numpy.ndarray
        The sizes of the valid (finite, positive) samples used in the fit.
    fitted : numpy.ndarray
        The fitted errors at ``sizes``, ready to plot over the data.
    """

    order: float
    constant: float
    sizes: np.ndarray
    fitted: np.ndarray


def convergence_order(sizes, errors) -> ConvergenceFit | None:
    """Fit ``error = constant * size**order`` in log-log space.

    Only valid samples, where both the size and the error are finite and positive, are used.

    Parameters
    ----------
    sizes : array_like of float
        Typically a resolution (points per cell, coarser to finer) or a step size (``dt``).
    errors : array_like of float
        The corresponding, necessarily positive, error norms, e.g. from :func:`error`.

    Returns
    -------
    ConvergenceFit or None
        The fit. ``order`` is negative when the error shrinks as ``sizes`` grows (e.g. more
        points per cell), and positive when it shrinks as ``sizes`` shrinks (e.g. a smaller
        ``dt``). ``None`` with fewer than two valid (finite, positive) samples.

    Examples
    --------
    >>> errors = [
    ...     run.evaluate("T").plasma.analysis.error(exact).isel(t=-1)
    ...     for run in runs
    ... ]
    >>> convergence_order([16, 32, 64], errors).order
    -2.01
    """
    sizes = np.asarray(sizes, dtype=float)
    errors = np.asarray(errors, dtype=float)
    valid = np.isfinite(sizes) & np.isfinite(errors) & (sizes > 0) & (errors > 0)
    if np.count_nonzero(valid) < 2:
        return None
    log_sizes, log_errors = np.log(sizes[valid]), np.log(errors[valid])
    order, intercept = np.polyfit(log_sizes, log_errors, 1)
    fitted = np.exp(intercept) * sizes[valid] ** order
    return ConvergenceFit(float(order), float(np.exp(intercept)), sizes[valid], fitted)


def growth_rate(data: xr.DataArray, fit: GrowthFit | None = None) -> FitResult | None:
    """Fit ``exp(rate*t + intercept)`` to a time series, using only finite, positive samples.

    The fit is a straight line through the logarithm of the samples within ``fit.window``.

    Parameters
    ----------
    data : xarray.DataArray
        The time series, with ``t`` as its only dimension (e.g. a field energy).
    fit : GrowthFit, optional
        The time window and whether the series is quadratic in the amplitude. Default:
        ``GrowthFit()``, every sample, the series itself.

    Returns
    -------
    FitResult or None
        The rate, the intercept, the times used and the fitted curve. ``None`` with fewer than
        two samples, or fewer than two finite, positive samples within the window.

    Raises
    ------
    ValueError
        If ``data`` has dimensions other than ``t``.

    See Also
    --------
    damping_rate : The same fit to the envelope of an oscillating series.

    Examples
    --------
    >>> energy = out.scalars["en_E"]
    >>> growth_rate(
    ...     energy, GrowthFit(window=(0.0, 5.0), amplitude_from_quadratic=True)
    ... ).rate
    """
    validate_array(data, required_dims=("t",))
    if data.dims != ("t",):
        raise ValueError(f"growth-rate input must have dims ('t',), got {data.dims}")
    fit = fit or GrowthFit()
    time, values = np.asarray(data.t), np.asarray(data)
    if len(time) < 2:
        return None
    lo = time[0] if fit.window[0] is None else fit.window[0]
    hi = time[-1] if fit.window[1] is None else fit.window[1]
    lo, hi = sorted((lo, hi))
    valid = (time >= lo) & (time <= hi) & np.isfinite(values) & (values > 0)
    if np.count_nonzero(valid) < 2:
        return None
    selected_time = time[valid]
    signal = (
        np.log(np.sqrt(values[valid]))
        if fit.amplitude_from_quadratic
        else np.log(values[valid])
    )
    rate, intercept = np.polyfit(selected_time, signal, 1)
    scale = 2.0 if fit.amplitude_from_quadratic else 1.0
    fitted = np.exp(scale * (rate * selected_time + intercept))
    return FitResult(float(rate), float(intercept), selected_time, fitted)


def envelope(data: xr.DataArray) -> xr.DataArray:
    """Local maxima of a time series: the interior samples not smaller than their neighbours.

    A sample is a peak when it is larger than the previous sample and not smaller than the next;
    the first and last samples never are.

    Parameters
    ----------
    data : xarray.DataArray
        The time series, with ``t`` as its only dimension.

    Returns
    -------
    xarray.DataArray
        The peaks of ``data``, a selection along ``t`` with the attributes and coordinates kept.

    Raises
    ------
    ValueError
        If ``data`` has dimensions other than ``t``.
    """
    validate_array(data, required_dims=("t",))
    if data.dims != ("t",):
        raise ValueError(f"envelope input must have dims ('t',), got {data.dims}")
    values = np.asarray(data)
    peak = np.zeros(len(values), dtype=bool)
    peak[1:-1] = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    return data.isel(t=np.flatnonzero(peak))


def damping_rate(data: xr.DataArray, fit: GrowthFit | None = None) -> FitResult | None:
    """Fit ``exp(rate*t + intercept)`` to the envelope of an oscillating time series.

    Use this for signals such as the field energy in Landau damping, where :func:`growth_rate` on
    the raw series would fit the oscillation. The envelope is the series' local maxima, see
    :func:`envelope`.

    Parameters
    ----------
    data : xarray.DataArray
        The oscillating time series, with ``t`` as its only dimension.
    fit : GrowthFit, optional
        ``fit.window`` restricts the peaks that are used; with ``fit.amplitude_from_quadratic``
        the rate of the amplitude is returned. Default: ``GrowthFit()``.

    Returns
    -------
    FitResult or None
        The fit to the peaks; the rate is negative for damping. ``None`` with fewer than two
        finite, positive peaks within the window.

    Raises
    ------
    ValueError
        If ``data`` has dimensions other than ``t``.

    See Also
    --------
    growth_rate, envelope

    Examples
    --------
    >>> damping_rate(
    ...     out.scalars["en_E"], GrowthFit(amplitude_from_quadratic=True)
    ... ).rate
    """
    return growth_rate(envelope(data), fit)


@dataclass(frozen=True)
class OscillationFit:
    """The frequency of an oscillating time series, from :func:`oscillation_frequency`.

    Attributes
    ----------
    omega : float
        The angular frequency ``2π / period``.
    period : float
        The period, from a straight-line fit through the times of successive zero crossings (half
        periods apart) or peaks (a period apart).
    times : numpy.ndarray
        The times of the crossings or peaks that were used.
    method : str
        ``"zero_crossings"`` or ``"peaks"``.
    """

    omega: float
    period: float
    times: np.ndarray
    method: str


def oscillation_frequency(
    data: xr.DataArray,
    *,
    window: tuple[float | None, float | None] = (None, None),
    method: str = "zero_crossings",
    detrend: bool = True,
) -> OscillationFit | None:
    """Measure the frequency of an oscillating time series from its zero crossings or its peaks.

    Zero crossings are unaffected by damping or growth, and are found to a fraction of a sample by
    linear interpolation; peaks are refined by a parabola through each peak and its neighbours.
    The period is the slope of a straight line through the crossing (or peak) times against their
    number, so a single late or early crossing hardly matters. A spectrum
    (:func:`~plasma_plots.spectral.spectral_peaks`) resolves several frequencies at once; this
    measures one, from a few periods, better than a frequency bin.

    Parameters
    ----------
    data : xarray.DataArray
        The time series, with ``t`` as its only dimension, e.g. a probe or a mode amplitude.
    window : (float or None, float or None), optional
        The time interval ``(t0, t1)`` used; ``None`` for an open end. Default: every sample.
    method : {"zero_crossings", "peaks"}, optional
        Count the crossings of the mean (half a period apart), or the maxima (a period apart; for a
        signal that does not cross its mean, e.g. an energy, whose peaks are half the field's
        period apart). Default: ``"zero_crossings"``.
    detrend : bool, optional
        Subtract the mean over the window first, so that crossings are of the mean. Default:
        ``True``.

    Returns
    -------
    OscillationFit or None
        The frequency, period and the times used; ``None`` with fewer than two crossings or peaks.

    Raises
    ------
    ValueError
        If ``data`` has dimensions other than ``t``, or ``method`` is unknown.

    See Also
    --------
    damping_rate : The decay of the same oscillation.

    Examples
    --------
    >>> oscillation_frequency(
    ...     phi.isel(eta1=8, eta2=0, eta3=0), window=(5.0, 40.0)
    ... ).omega
    """
    validate_array(data, required_dims=("t",))
    if data.dims != ("t",):
        raise ValueError(f"oscillation_frequency needs dims ('t',), got {data.dims}")
    if method not in ("zero_crossings", "peaks"):
        raise ValueError(
            f"unknown method {method!r}; expected 'zero_crossings' or 'peaks'"
        )
    t0, t1 = window
    t = np.asarray(data.t, dtype=float)
    keep = (
        np.isfinite(np.asarray(data, dtype=float))
        & (t >= (-np.inf if t0 is None else t0))
        & (t <= (np.inf if t1 is None else t1))
    )
    t, values = t[keep], np.asarray(data, dtype=float)[keep]
    if detrend and values.size:
        values = values - values.mean()
    if method == "zero_crossings":
        sign = np.signbit(values)
        i = np.flatnonzero(sign[:-1] != sign[1:])
        # linear interpolation between the two samples around each crossing
        times = t[i] - values[i] * (t[i + 1] - t[i]) / (values[i + 1] - values[i])
        spacing = 0.5  # crossings are half a period apart
    else:
        i = (
            np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:]))
            + 1
        )
        left, mid, right = values[i - 1], values[i], values[i + 1]
        curvature = left - 2 * mid + right
        shift = np.where(
            curvature != 0,
            0.5 * (left - right) / np.where(curvature != 0, curvature, 1),
            0.0,
        )
        times = t[i] + shift * 0.5 * (t[i + 1] - t[i - 1])
        spacing = 1.0
    if times.size < 2:
        return None
    period = (
        float(np.polyfit(np.arange(times.size), times, 1)[0]) / spacing
    )  # the slope: time per event
    return OscillationFit(
        omega=2 * np.pi / period, period=period, times=times, method=method
    )


def norm(data: xr.DataArray, *, dims=None, squared: bool = False) -> xr.DataArray:
    """L2 norm over ``dims`` (default: every dimension except ``t``), as a function of the rest.

    The discrete norm ``√(Σ f²)``, a plain sum over the grid points: neither divided by their
    number nor weighted by the volume element (see :func:`error` and :func:`volume_integral`
    for those).

    Parameters
    ----------
    data : xarray.DataArray
        The field.
    dims : str or sequence of str, optional
        The dimensions summed over. Default: every dimension except ``t``.
    squared : bool, optional
        Return the squared norm ``Σ f²`` instead. Default: ``False``.

    Returns
    -------
    xarray.DataArray
        The norm as a function of the remaining dimensions, labeled ``"norm of ..."`` (or
        ``"squared norm of ..."``), keeping the run provenance attributes.

    Examples
    --------
    >>> div_B.plasma.analysis.norm().plasma.plot.timeseries()
    """
    validate_array(data)
    dims = [dim for dim in data.dims if dim != "t"] if dims is None else list(dims)
    total = (data**2).sum(dims)
    out = total if squared else np.sqrt(total)
    out.attrs = {
        key: value for key, value in data.attrs.items() if key in ("run", "run_name")
    }
    label = _label(data)
    out.attrs["label"] = (
        f"squared norm of {label}".strip() if squared else f"norm of {label}".strip()
    )
    return out


def drift(data: xr.DataArray, *, ref=None) -> xr.DataArray:
    """Signed deviation from an explicit reference or the first time sample.

    Parameters
    ----------
    data : xarray.DataArray
        The array, with a ``t`` dimension.
    ref : float or array_like or xarray.DataArray, optional
        The reference, broadcast against ``data``. Default: ``data`` at the first time sample.

    Returns
    -------
    xarray.DataArray
        ``data - ref``, with the attributes of ``data`` and the label ``"... drift"``.

    See Also
    --------
    relative_error : The absolute relative deviation.

    Examples
    --------
    >>> drift(out.scalars["en_tot"]).plasma.plot.timeseries()
    """
    validate_array(data, required_dims=("t",))
    reference = data.isel(t=0) if ref is None else ref
    out = data - reference
    out.attrs = dict(data.attrs)
    out.attrs["label"] = f"{_label(data)} drift".strip()
    return out


VELOCITY_DIMS = ("v1", "v2", "v3")


def _provenance(data: xr.DataArray) -> dict:
    return {
        key: value for key, value in data.attrs.items() if key in ("run", "run_name")
    }


def _select_dims(data: xr.DataArray, dims, default) -> list[str]:
    if dims is None:
        selected = [dim for dim in default if dim in data.dims]
        if not selected:
            raise ValueError(
                f"{data.name!r} has none of the dimensions {default}; its dimensions are {data.dims}"
            )
        return selected
    selected = [dims] if isinstance(dims, str) else list(dims)
    missing = [dim for dim in selected if dim not in data.dims]
    if missing:
        raise ValueError(
            f"{data.name!r} has no dimensions {missing}; its dimensions are {data.dims}"
        )
    return selected


def spatial_average(data: xr.DataArray, *, dims=None) -> xr.DataArray:
    """Mean over the logical space dimensions, e.g. a binned f(t, eta1, v1) becomes f(t, v1).

    The mean is uniform in the logical coordinates, which is the volume average on a Cartesian
    domain; on a mapped domain it is not weighted by the Jacobian (use :func:`volume_integral`
    for that). Physical ``X``, ``Y``, ``Z`` coordinates that depend on the averaged dimensions
    are dropped.

    Parameters
    ----------
    data : xarray.DataArray
        The array, e.g. a binned distribution function.
    dims : str or sequence of str, optional
        The dimensions averaged over. Default: every logical dimension (``eta1``, ``eta2``,
        ``eta3``, or GVEC's ``rho``, ``theta``, ``zeta``) that ``data`` has.

    Returns
    -------
    xarray.DataArray
        The mean over ``dims``, as a function of the remaining dimensions, with the attributes
        of ``data`` and the label ``"average of ..."``.

    Raises
    ------
    ValueError
        If ``data`` has none of the default dimensions, or lacks one of ``dims``.
    """
    validate_array(data)
    averaged = _select_dims(data, dims, logical_dims(data))
    out = data.mean(averaged, keep_attrs=True)
    out.attrs["label"] = f"average of {_label(data)}".strip()
    out.attrs.pop("long_name", None)
    return out


def _bin_widths(data: xr.DataArray, dim: str) -> xr.DataArray:
    coordinate = np.asarray(data.coords[dim]) if dim in data.coords else None
    if coordinate is None or len(coordinate) < 2:
        raise ValueError(f"dimension {dim!r} needs a coordinate with at least two bins")
    return xr.DataArray(
        np.gradient(coordinate), dims=(dim,), coords={dim: data.coords[dim]}
    )


def velocity_moments(f: xr.DataArray, *, dims=None) -> xr.Dataset:
    """Moments of a binned distribution function over its velocity dimensions.

    The moments are functions of the remaining dimensions, for example ``(t, eta1)`` for an
    ``e1_v1`` product. The integrals are sums over the bins, weighted by the bin widths (from
    ``numpy.gradient`` of the velocity coordinates). The values keep the normalization of the
    run; see ``Output.to_si``.

    Parameters
    ----------
    f : xarray.DataArray
        The binned distribution function. A product named ``delta_f`` has only the density,
        which is then the density perturbation, because its mean and variance are not defined.
    dims : str or sequence of str, optional
        The velocity dimensions integrated over, each with a coordinate of at least two bins.
        Default: every one of ``v1``, ``v2``, ``v3`` that ``f`` has.

    Returns
    -------
    xarray.Dataset
        The moments, over the remaining dimensions:

        * ``density``: the zeroth moment, ``n = ∫ f dv``.
        * ``mean_<dim>``: the mean velocity ``u = ∫ v f dv / n`` along each dimension.
        * ``variance_<dim>``: ``∫ (v − u)² f dv / n``. In normalized units this is the
          temperature over the particle mass along that direction, ``T/m``.

        Where the density is not positive, the mean and variance are NaN.

    Raises
    ------
    ValueError
        If ``f`` has none of the default dimensions, lacks one of ``dims``, or a dimension has
        fewer than two bins.

    Examples
    --------
    >>> moments = velocity_moments(f)  # f(t, eta1, v1)
    >>> moments.variance_v1.isel(t=-1).plasma.plot.lineout()
    """
    validate_array(f)
    integrated = _select_dims(f, dims, VELOCITY_DIMS)
    volume = 1.0
    for dim in integrated:
        volume = volume * _bin_widths(f, dim)
    density = (f * volume).sum(integrated)

    label = _label(f)
    variables = {"density": (density, f"density of {label}")}
    if f.name != "delta_f":
        weight = density.where(density > 0)
        for dim in integrated:
            mean = (f * f[dim] * volume).sum(integrated) / weight
            variance = (f * (f[dim] - mean) ** 2 * volume).sum(integrated) / weight
            variables[f"mean_{dim}"] = (mean, f"mean {dim}")
            variables[f"variance_{dim}"] = (variance, f"variance of {dim}")

    provenance = _provenance(f)
    out = {}
    for name, (values, description) in variables.items():
        values.attrs = {**provenance, "label": description.strip()}
        out[name] = values.rename(name)
    return xr.Dataset(out, attrs=provenance)


def relative_error(data: xr.DataArray, *, ref=None, skip_first=True) -> xr.DataArray:
    """Absolute relative deviation from an explicit reference or first sample.

    Parameters
    ----------
    data : xarray.DataArray
        The array, with a ``t`` dimension, e.g. a conserved energy.
    ref : float or array_like or xarray.DataArray, optional
        The reference, broadcast against ``data``; must be non-zero everywhere. Default: ``data``
        at the first time sample.
    skip_first : bool, optional
        Leave out the first time sample (zero against the default reference). Default: ``True``.

    Returns
    -------
    xarray.DataArray
        ``|data − ref| / |ref|``, labeled ``"relative error of ..."`` with empty units.

    Raises
    ------
    ValueError
        If the reference is zero anywhere.

    See Also
    --------
    drift : The signed deviation.

    Examples
    --------
    >>> relative_error(out.scalars["en_tot"]).plasma.plot.timeseries()
    """
    validate_array(data, required_dims=("t",))
    reference = data.isel(t=0) if ref is None else ref
    if np.any(np.asarray(reference) == 0):
        raise ValueError("cannot take a relative error against a reference of zero")
    out = abs(data - reference) / abs(reference)
    out.attrs = {
        key: value for key, value in data.attrs.items() if key in ("run", "run_name")
    }
    out.attrs.update(label=f"relative error of {_label(data)}".strip(), units="")
    return out.isel(t=slice(1, None)) if skip_first else out


ORBIT_CLASSES = {0: "passing", 1: "trapped", -1: "lost"}


def classify_orbits(orbits: xr.Dataset, *, v_par: str = "v_par") -> xr.DataArray:
    """Classify each marker of an orbits product as passing (0), trapped (1) or lost (-1).

    The same criteria as Struphy's ``post_process_orbit_classification``: a marker is trapped if
    its parallel velocity ``v_par`` ever has the opposite sign to its initial one, and lost if at
    any saved time every quantity is zero (how Struphy stores a marker that has left the domain).
    Lost takes precedence over trapped.

    Parameters
    ----------
    orbits : xarray.Dataset
        The orbits product, with variables over ``(t, marker)``.
    v_par : str, optional
        The name of the parallel-velocity variable. Default: ``"v_par"``.

    Returns
    -------
    xarray.DataArray
        A ``(marker,)`` array of integer codes named ``classification``; the names are in
        ``attrs["flag_meanings"]`` and in :data:`ORBIT_CLASSES`.

    Raises
    ------
    ValueError
        If ``orbits`` has no variable ``v_par``, or it is not over ``(t, marker)``.

    See Also
    --------
    bounce_period, orbit_invariants

    Examples
    --------
    >>> codes = classify_orbits(orbits)
    >>> trapped = orbits.sel(marker=codes == 1)
    """
    if v_par not in orbits.data_vars:
        raise ValueError(
            f"orbit classification needs the parallel velocity {v_par!r}; this dataset has {tuple(orbits.data_vars)}"
        )
    if set(orbits[v_par].dims) != {"t", "marker"}:
        raise ValueError(
            f"{v_par!r} must have dims ('t', 'marker'); got {orbits[v_par].dims}"
        )
    velocity = orbits[v_par].transpose("t", "marker")
    trapped = (velocity * velocity.isel(t=0) < 0).any("t")
    all_zero = velocity == 0
    for name in orbits.data_vars:
        if set(orbits[name].dims) == {"t", "marker"}:
            all_zero = all_zero & (orbits[name] == 0)
    lost = all_zero.any("t")
    codes = xr.where(lost, -1, xr.where(trapped, 1, 0)).astype(int)
    codes.name = "classification"
    codes.attrs = {
        **{
            key: value
            for key, value in orbits.attrs.items()
            if key in ("run", "run_name")
        },
        "label": "orbit classification",
        "flag_values": list(ORBIT_CLASSES),
        "flag_meanings": " ".join(ORBIT_CLASSES.values()),
    }
    return codes


@dataclass(frozen=True)
class BranchFit:
    """A dispersion branch fitted as ``omega = velocity * k``, from :func:`fit_dispersion_branches`.

    Attributes
    ----------
    velocity : float
        The fitted slope, the phase velocity of the branch.
    k : numpy.ndarray
        The wavenumbers of the ridge points used in the fit.
    omega : numpy.ndarray
        The angular frequencies of the ridge points, one per ``k``.
    """

    velocity: float
    k: np.ndarray
    omega: np.ndarray


def _local_maxima(column: np.ndarray, order: int) -> np.ndarray:
    """Indices where ``column`` exceeds every one of its ``order`` neighbors on both sides.

    Out-of-range neighbors are clipped to the nearest edge sample, so an edge index is never a
    maximum unless the whole column is flat there (matching ``scipy.signal.argrelextrema``'s
    default ``mode="clip"``).
    """
    n = column.size
    index = np.arange(n)
    is_max = np.ones(n, dtype=bool)
    for shift in range(1, order + 1):
        is_max &= column > column[np.clip(index - shift, 0, n - 1)]
        is_max &= column > column[np.clip(index + shift, 0, n - 1)]
    return np.flatnonzero(is_max)


def fit_dispersion_branches(
    spectrum: xr.DataArray,
    *,
    n_branches: int,
    k_range: tuple[float, float] | None = None,
    noise_level: float = 0.5,
    order: int = 10,
) -> list[BranchFit]:
    """Fit ``omega = v * k`` to each of ``n_branches`` straight ridges in an (omega, k) power spectrum.

    The spectrum is as returned by :func:`power_spectrum`, and there is no theoretical curve to
    guide the search -- useful when several linear wave branches are excited at once and there is
    nothing to search around yet. (For a single, possibly non-linear branch with a known
    theoretical curve to guide the search instead, take the frequency of maximum power in a window
    of ``spectrum`` around that curve at each k, rather than this blind approach.)

    Only non-negative ``omega`` and ``k`` are scanned (a real signal's spectrum is symmetric under
    ``(k, omega) -> (-k, -omega)``, so every branch already appears on both sides of ``k = 0``). At
    each remaining k, the local maxima of the spectrum along omega are found; a k column
    contributes to the fit only where it has exactly ``n_branches`` maxima above ``noise_level``
    times that column's own peak power, taken in increasing-omega order.

    Parameters
    ----------
    spectrum : xarray.DataArray
        The power spectrum, with dimensions ``omega`` and ``k``.
    n_branches : int
        The number of branches, at least 1.
    k_range : (float, float), optional
        The interval of non-negative k's that are scanned. Default: ``(k.max() / 8, k.max() / 2)``,
        which in practice skips both the low-k region where branches have not yet separated, and
        the folded Nyquist edge.
    noise_level : float, optional
        Maxima count only above this fraction of the column's peak power. Default: 0.5.
    order : int, optional
        A local maximum must exceed every one of its ``order`` neighbors on both sides along
        omega (out-of-range neighbors are clipped to the edge sample, as in
        ``scipy.signal.argrelextrema``). Default: 10.

    Returns
    -------
    list of BranchFit
        One fit per branch, in increasing-omega order at the low-k end of ``k_range``;
        ``.velocity`` is the fitted slope, ``.k``/``.omega`` the ridge points used.

    Raises
    ------
    ValueError
        If ``spectrum`` lacks the ``omega`` or ``k`` dimension, ``n_branches`` is not positive, or
        no k in ``k_range`` has exactly ``n_branches`` peaks above the noise level.

    See Also
    --------
    power_spectrum

    Examples
    --------
    >>> spectrum = power_spectrum(b.isel(eta2=0, eta3=0, component=0))
    >>> [fit.velocity for fit in fit_dispersion_branches(spectrum, n_branches=2)]
    """
    if not {"omega", "k"} <= set(spectrum.dims):
        raise ValueError(
            f"spectrum must have dims 'omega' and 'k'; got {spectrum.dims}"
        )
    if n_branches < 1:
        raise ValueError("n_branches must be positive")

    omega_mask, k_mask = spectrum.omega.values >= 0, spectrum.k.values >= 0
    omega, k = spectrum.omega.values[omega_mask], spectrum.k.values[k_mask]
    power = np.asarray(spectrum.transpose("omega", "k"))[np.ix_(omega_mask, k_mask)]

    if k_range is None:
        k_range = (k.max() / 8.0, k.max() / 2.0)
    lo, hi = k_range
    scan = np.flatnonzero((k >= lo) & (k <= hi))

    k_fit: list[float] = []
    peaks_fit: list[list[float]] = [[] for _ in range(n_branches)]
    for i in scan:
        column = power[:, i]
        maxima = _local_maxima(column, order)
        peaks = sorted(m for m in maxima if column[m] > noise_level * column.max())
        if len(peaks) != n_branches:
            continue
        k_fit.append(k[i])
        for branch, m in zip(peaks_fit, peaks):
            branch.append(omega[m])

    if not k_fit:
        raise ValueError(
            f"no k in {tuple(k_range)} has exactly {n_branches} peaks above "
            f"noise_level={noise_level}; try a different k_range, noise_level, or order"
        )
    k_fit = np.asarray(k_fit)
    return [
        BranchFit(
            float(np.polyfit(k_fit, np.asarray(branch), deg=1)[0]),
            k_fit,
            np.asarray(branch),
        )
        for branch in peaks_fit
    ]


def power_spectrum(
    data: xr.DataArray, *, dim: str | None = None, detrend: bool = True
) -> xr.DataArray:
    """The 2-D power spectrum of a ``(t, dim)`` signal, as a function of frequency and wavenumber.

    A plain space-time FFT, as a function of angular frequency and wavenumber -- the basis of a
    dispersion-relation plot (:meth:`~plasma_plots.accessors.ArrayPlots.dispersion`),
    independent of Struphy. Built on :func:`plasma_plots.spectral.fft`, so it shares its
    conventions: coefficients are divided by the sample counts, and the power sums to the mean
    square of the signal.

    Parameters
    ----------
    data : xarray.DataArray
        The signal, with exactly the two dimensions ``t`` and ``dim``, each on a uniform grid.
    dim : str, optional
        The spatial dimension. Default: the sole dimension other than ``t``.
    detrend : bool, optional
        Remove the time-mean at each point of ``dim`` first, which otherwise dominates the
        spectrum as a spurious zero-frequency line. Default: ``True``.

    Returns
    -------
    xarray.DataArray
        The power ``|ĉ|²``, named ``power``, over ``(omega, k)``.

    Raises
    ------
    ValueError
        If ``dim`` is not given and ``data`` has more than one dimension besides ``t``, or
        ``data`` has dimensions other than ``t`` and ``dim``.

    See Also
    --------
    fit_dispersion_branches

    Examples
    --------
    >>> spectrum = power_spectrum(phi.isel(eta2=0, eta3=0), dim="eta1")
    """
    validate_array(data, required_dims=("t",))
    others = [d for d in data.dims if d != "t"]
    if dim is None:
        if len(others) != 1:
            raise ValueError(
                f"dim is required unless data has exactly one dimension besides 't'; got {data.dims}"
            )
        dim = others[0]
    elif dim not in data.dims:
        raise ValueError(
            f"{dim!r} is not a dimension of this array; its dimensions are {data.dims}"
        )
    if set(data.dims) != {"t", dim}:
        raise ValueError(
            f"select every dimension except 't' and {dim!r} first; got {data.dims}"
        )
    from .spectral import fft

    signal = data.transpose("t", dim).reset_coords(drop=True)
    coefficients = fft(fft(signal, dim="t", detrend=detrend), dim=dim)
    spectrum = (
        (abs(coefficients) ** 2).rename({f"k_{dim}": "k"}).transpose("omega", "k")
    )
    spectrum.name = "power"
    spectrum.attrs = {"label": f"power spectrum of {_label(data)}".strip(), "units": ""}
    return spectrum


# ---------------------------------------------------------------------------------------------
# Volume integrals and field energies (after struphy's TAE_example_Shrut field_energies.py)
# ---------------------------------------------------------------------------------------------


def quadrature_weights(coordinate, *, period: float | None = None) -> np.ndarray:
    """Quadrature weights for samples of a logical coordinate in ``[0, 1]``, or of an angle.

    Struphy's cell centers (uniform, half a cell from each end) get the midpoint rule, which
    integrates over the whole unit interval; a uniform grid over one full ``period`` (an angle,
    e.g. GVEC's) the rectangle rule, exact for its Fourier modes; any other grid the trapezoidal
    rule over the sampled range. A single point (a 2-D run's flat direction) has weight 1.

    Parameters
    ----------
    coordinate : array_like of float
        The sample points along one logical direction.
    period : float, optional
        The direction's period, if it is an angle (see :func:`plasma_plots.arrays.angle_period`).
        Default: none.

    Returns
    -------
    numpy.ndarray
        One weight per sample point.

    See Also
    --------
    volume_integral, field_energy
    """
    x = np.asarray(coordinate, dtype=float)
    if x.size == 1:
        return np.ones(1)
    h = np.diff(x)
    if (
        np.allclose(h, h[0])
        and np.isclose(x[0], h[0] / 2)
        and np.isclose(x[-1], 1 - h[0] / 2)
    ):
        return np.full(x.size, h[0])
    if (
        period is not None
        and np.allclose(h, h[0])
        and np.isclose(x.size * h[0], period)
    ):
        return np.full(x.size, h[0])
    weights = np.empty(x.size)
    weights[1:-1] = (x[2:] - x[:-2]) / 2
    weights[0], weights[-1] = h[0] / 2, h[-1] / 2
    return weights


def _geometry(data: xr.DataArray, domain=None):
    """``|sqrt g|`` and the metric ``G`` (3, 3, n1, n2, n3) on the logical grid of ``data``: from a
    struphy ``domain`` (exact), or from the Jacobian of the attached X, Y, Z coordinates.
    """
    spatial = logical_dims(data)
    missing = [d for d in spatial if d not in data.dims]
    if missing:
        raise ValueError(
            f"integrals need the logical dimensions {spatial}; {missing} are missing"
        )
    if domain is not None:
        etas = [np.asarray(data[d], dtype=float) for d in spatial]
        sqrt_g = np.abs(np.asarray(domain.jacobian_det(*etas), dtype=float))
        metric = np.asarray(domain.metric(*etas), dtype=float)
        return sqrt_g.reshape([data.sizes[d] for d in spatial]), metric
    from .arrays import mapping_jacobian

    jacobian = mapping_jacobian(data.transpose(..., *spatial))
    sqrt_g = np.abs(np.linalg.det(np.moveaxis(jacobian, (0, 1), (-2, -1))))
    metric = np.einsum("ai...,aj...->ij...", jacobian, jacobian)
    return sqrt_g, metric


def _weights(data: xr.DataArray, dim: str, given=None) -> np.ndarray:
    """The quadrature weights along ``dim``: ``given``, else a ``<dim>_weight`` coordinate (GVEC's
    Gauss weights, one per point or one for all), else :func:`quadrature_weights`."""
    if given is None and f"{dim}_weight" in data.coords:
        given = np.broadcast_to(
            np.asarray(data.coords[f"{dim}_weight"], dtype=float), (data.sizes[dim],)
        )
    if given is None:
        return quadrature_weights(data[dim], period=angle_period(data, dim))
    values = np.asarray(given, dtype=float)
    if values.size != data.sizes[dim]:
        raise ValueError(
            f"{values.size} quadrature weights for {data.sizes[dim]} points along {dim!r}"
        )
    return values


def _integrate(integrand: xr.DataArray, quadrature=None, dims=None) -> xr.DataArray:
    dims = logical_dims(integrand) if dims is None else dims
    weights = 1.0
    for dim in dims:
        weights = weights * xr.DataArray(
            _weights(integrand, dim, (quadrature or {}).get(dim)), dims=(dim,)
        )
    weight_coords = [f"{d}_weight" for d in dims if f"{d}_weight" in integrand.coords]
    return (
        (integrand * weights).sum(list(dims)).drop_vars(weight_coords, errors="ignore")
    )


def _spatial_array(values, data):
    spatial = logical_dims(data)
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=spatial,
        coords={d: data[d] for d in spatial},
    )


def volume_integral(
    data: xr.DataArray,
    *,
    form: int = 0,
    weight=None,
    domain=None,
    quadrature=None,
    jacobian=None,
) -> xr.DataArray:
    """``∫ w f dV`` over the logical grid, as a function of every other dimension (e.g. ``t``).

    A function (``form=0``) is integrated with the volume element ``|√g| dη``; a density (a
    3-form, e.g. Struphy's L2 fields, ``form=3``) as ``∫ f dη``.

    Parameters
    ----------
    data : xarray.DataArray
        The integrand, with the three logical dimensions (``eta1``, ``eta2``, ``eta3``, or GVEC's
        ``rho``, ``theta``, ``zeta``; see :func:`plasma_plots.arrays.logical_dims`).
    form : {0, 3}, optional
        ``0`` (default) for a function, ``3`` for a density.
    weight : array_like, optional
        An extra weight ``w`` over the logical grid, broadcast against ``data``.
    domain : struphy domain, optional
        The mapping (``out.domain``), which gives the exact ``|√g|``. Without one, ``|√g|`` comes
        from the numerical Jacobian of the ``X``, ``Y``, ``Z`` coordinates (see
        :func:`plasma_plots.arrays.mapping_jacobian`). Only needed for ``form=0``.
    quadrature : dict, optional
        Maps logical dimensions to explicit weights, one per point (e.g. Gauss weights, see
        ``out.analysis.quadrature_grid()``). Directions left out take a ``<dim>_weight``
        coordinate (GVEC's integration points, see :func:`plasma_plots.gvec.from_gvec`), else
        :func:`quadrature_weights`: the midpoint rule on Struphy's cell centers, the rectangle rule
        over a full period of an angle, else trapezoidal.
    jacobian : xarray.DataArray, optional
        The Jacobian determinant ``√g`` on the grid, e.g. GVEC's ``Jac`` (its absolute value is
        used), instead of ``domain`` or the numerical one. Only needed for ``form=0``.

    Returns
    -------
    xarray.DataArray
        The integral, as a function of the dimensions other than the logical ones, labeled
        ``"integral of ..."``. On GVEC's grid over one field period, multiply by ``nfp`` for the
        whole device.

    Raises
    ------
    ValueError
        If ``form`` is not 0 or 3, a logical dimension is missing, or a quadrature has the wrong
        number of weights.

    See Also
    --------
    field_energy, quadrature_weights

    Examples
    --------
    >>> total_charge = volume_integral(rho, domain=out.domain)
    >>> mass = volume_integral(n3, form=3)  # a 3-form: no |√g|
    >>> # GVEC
    >>> volume = volume_integral(xr.ones_like(ev.mod_B), jacobian=ev.Jac) * ev.nfp
    """
    if form not in (0, 3):
        raise ValueError(
            "volume_integral takes form=0 (a function) or form=3 (a density)"
        )
    validate_array(data)
    integrand = data
    if form == 0 and jacobian is not None:
        integrand = integrand * abs(jacobian)
    elif form == 0:
        sqrt_g, _ = _geometry(data, domain)
        integrand = integrand * _spatial_array(sqrt_g, data)
    if weight is not None:
        integrand = integrand * np.asarray(weight, dtype=float)
    out = _integrate(integrand, quadrature)
    out.attrs = {**_provenance(data), "label": f"integral of {_label(data)}".strip()}
    return out


def surface_average(
    data: xr.DataArray, *, jacobian=None, domain=None, quadrature=None
) -> xr.DataArray:
    """The flux-surface average ``⟨f⟩ = ∫ f √g dθ dζ / ∫ √g dθ dζ`` over the two angles.

    The angles are the second and third logical dimensions (``eta2``, ``eta3``, or GVEC's
    ``theta``, ``zeta`` or Boozer or PEST angles; see :func:`plasma_plots.arrays.logical_dims`),
    and the average is a function of the radius and every other dimension. The angles are
    integrated as in :func:`volume_integral`: GVEC's Gauss weights where given, else the
    rectangle rule over a full period. On the magnetic axis, where ``√g`` vanishes, it is the
    plain mean over the angles.

    Parameters
    ----------
    data : xarray.DataArray
        The field, over the radial and both angular logical dimensions.
    jacobian : xarray.DataArray, optional
        ``√g`` on the same grid, e.g. GVEC's ``Jac`` (its absolute value is used). Default: from
        ``domain``, else the numerical Jacobian of the ``X``, ``Y``, ``Z`` coordinates, which needs
        at least two radial points.
    domain : struphy domain, optional
        The mapping, for the exact ``√g`` of a Struphy run.
    quadrature : dict, optional
        Explicit weights for the angles, as for :func:`volume_integral`.

    Returns
    -------
    xarray.DataArray
        ``⟨f⟩`` as a function of the radius (and every non-spatial dimension), labeled
        ``"⟨...⟩"``.

    Raises
    ------
    ValueError
        If an angle is missing, or ``√g`` can't be computed (no ``jacobian``, ``domain`` or
        ``X``, ``Y``, ``Z`` on at least two radial points).

    See Also
    --------
    volume_integral

    Examples
    --------
    >>> # GVEC: ⟨|B|⟩(rho)
    >>> ev.mod_B.plasma.analysis.surface_average(jacobian=ev.Jac)
    >>> surface_average(p, domain=out.domain)  # Struphy
    """
    validate_array(data)
    _, poloidal, toroidal = logical_dims(data)
    missing = [d for d in (poloidal, toroidal) if d not in data.dims]
    if missing:
        raise ValueError(
            f"a surface average needs the angles {poloidal!r}, {toroidal!r}; {missing} are missing"
        )
    if jacobian is not None:
        sqrt_g = abs(jacobian)
    else:
        values, _ = _geometry(data, domain)
        sqrt_g = _spatial_array(values, data)
    weights = xr.ones_like(sqrt_g)
    for dim in (poloidal, toroidal):
        weights = weights * xr.DataArray(
            _weights(data, dim, (quadrature or {}).get(dim)), dims=(dim,)
        )
    measure = (sqrt_g * weights).sum((poloidal, toroidal))
    plain = (data * weights).sum((poloidal, toroidal)) / weights.sum(
        (poloidal, toroidal)
    )
    out = (
        (data * sqrt_g * weights).sum((poloidal, toroidal)) / measure.where(measure > 0)
    ).fillna(plain)
    drop = [f"{d}_weight" for d in (poloidal, toroidal) if f"{d}_weight" in out.coords]
    out = out.drop_vars(drop)
    out.attrs = {**_provenance(data), "label": f"⟨{_label(data)}⟩"}
    for name in ("units", "nfp"):  # an average keeps the units
        if name in data.attrs:
            out.attrs[name] = data.attrs[name]
    return out


def rational_surfaces(
    profile: xr.DataArray,
    *,
    count: int = 4,
    nfp: int | None = None,
    max_denominator: int = 12,
) -> xr.DataArray:
    """Where a rotational transform (or safety factor) profile takes low-order rational values.

    The values are the fractions ``n/m`` with ``m ≤ max_denominator`` and ``n`` a multiple of
    ``nfp`` (for ι = n/m, the resonances of a stellarator with nfp field periods) that the profile
    reaches; the ``count`` of lowest order (smallest ``m``, then ``|n|``) are kept. Each crossing
    is found by linear interpolation between samples, so a non-monotonic profile gives several.

    Parameters
    ----------
    profile : xarray.DataArray
        A 1-D profile, e.g. ``iota`` over ``rho``.
    count : int, optional
        The number of rational values. Default: 4.
    nfp : int, optional
        The numerators are multiples of it. Default: the profile's ``nfp`` attribute (GVEC's, see
        :func:`plasma_plots.gvec.from_gvec`), else 1.
    max_denominator : int, optional
        The largest ``m``. Default: 12.

    Returns
    -------
    xarray.DataArray
        The positions of the crossings (in the profile's coordinate), over a ``surface``
        dimension with the coordinates ``n``, ``m`` and ``value`` (``n/m``), lowest order first;
        empty if the profile reaches none.

    Raises
    ------
    ValueError
        If ``profile`` is not one-dimensional.

    See Also
    --------
    plasma_plots.plotting.plot_lineout : ``rationals=`` marks them on the profile.

    Examples
    --------
    >>> rational_surfaces(ev.iota, count=3)
    >>> ev.iota.plasma.plot.lineout(rationals=3)
    """
    from math import gcd

    validate_array(profile)
    if profile.ndim != 1:
        raise ValueError(
            f"rational surfaces need a 1-D profile; {profile.name!r} has dims {profile.dims}"
        )
    dim = profile.dims[0]
    nfp = int(profile.attrs.get("nfp", 1) if nfp is None else nfp)
    x = np.asarray(profile[dim], dtype=float)
    y = np.asarray(profile, dtype=float)
    finite = np.isfinite(y)
    lo, hi = float(y[finite].min()), float(y[finite].max())
    candidates = []
    for m in range(1, max_denominator + 1):
        for n in range(
            int(np.ceil(lo * m / nfp)) * nfp, int(np.floor(hi * m / nfp)) * nfp + 1, nfp
        ):
            if gcd(abs(n), m) == 1 or (n == 0 and m == 1):
                candidates.append((m, abs(n), n))
    candidates = sorted(set(candidates))[:count]
    rows = []
    for m, _, n in candidates:
        value = n / m
        d = y - value
        for i in range(len(x) - 1):
            if not (np.isfinite(d[i]) and np.isfinite(d[i + 1])):
                continue
            if d[i] == 0:
                rows.append((n, m, value, x[i]))
            elif d[i] * d[i + 1] < 0:
                rows.append(
                    (n, m, value, x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i]))
                )
        if d[-1] == 0:
            rows.append((n, m, value, x[-1]))
    n, m, value, where = (
        (np.array(column) for column in zip(*rows)) if rows else ([], [], [], [])
    )
    out = xr.DataArray(
        np.asarray(where, dtype=float),
        dims="surface",
        coords={
            "n": ("surface", np.asarray(n, dtype=int)),
            "m": ("surface", np.asarray(m, dtype=int)),
            "value": ("surface", np.asarray(value, dtype=float)),
        },
        name=dim,
        attrs={"label": f"rational surfaces of {_label(profile)}"},
    )
    return out


def field_energy(
    data: xr.DataArray,
    *,
    form: int | str | None = None,
    weight=None,
    domain=None,
    normalization: float = 1.0,
    quadrature=None,
) -> xr.DataArray:
    r"""The quadratic energy ``α ½ ∫ w ωᵀ A ω dη``, as a function of time.

    ``form`` says what ``data`` holds, and sets the metric factor ``A`` (as struphy's mass
    matrices do, so that e.g. LinearMHD's ``en_U`` is ``field_energy(u, form=2, weight=n0)``):

    ========================  ===========================================  ==============
    ``form``                  data                                         ``A``
    ========================  ===========================================  ==============
    ``None`` (default)        a function, or Cartesian vector components    ``|√g|``
    ``0``                     0-form                                        ``|√g|``
    ``1``                     1-form components                             ``G⁻¹ |√g|``
    ``2``                     2-form components                             ``G / |√g|``
    ``3``                     3-form                                        ``1 / |√g|``
    ``"v"``                   contravariant vector components               ``G |√g|``
    ========================  ===========================================  ==============

    Here ``G = Jᵀ J`` is the metric of the mapping and ``|√g|`` its Jacobian determinant. The
    energy of a filtered field, e.g. from :func:`~plasma_plots.spectral.filter_time`, measures
    how much of the energy is in that mode. A spline field squared is integrated exactly only
    with enough points per element: evaluate it at Gauss points for accurate energies.

    Parameters
    ----------
    data : xarray.DataArray
        The field, with the dimensions ``eta1``, ``eta2``, ``eta3``; vectors have a ``component``
        dimension of size 3.
    form : {None, 0, 1, 2, 3, "v"}, optional
        What ``data`` holds, see the table. Default: ``None``, a function or Cartesian vector
        components.
    weight : array_like, optional
        The weight ``w`` over ``(eta1, eta2, eta3)`` (e.g. a background density ``n0``).
        Non-finite values (e.g. ``1/p0`` where ``p0`` vanishes on the boundary) are left out.
    domain : struphy domain, optional
        The mapping (``out.domain``), which gives the exact ``|√g|`` and ``G``. Without one, they
        come from the numerical Jacobian of the ``X``, ``Y``, ``Z`` coordinates (see
        :func:`plasma_plots.arrays.mapping_jacobian`).
    normalization : float, optional
        The prefactor ``α``. Default: 1.
    quadrature : dict, optional
        Explicit quadrature weights per logical dimension, as for :func:`volume_integral`.

    Returns
    -------
    xarray.DataArray
        The energy, as a function of the dimensions other than ``eta1``, ``eta2``, ``eta3`` and
        ``component`` (typically ``t``), labeled ``"energy of ..."``.

    Raises
    ------
    ValueError
        If ``form`` is not one of the above or does not match ``data`` (a vector form for a
        scalar or the reverse), a vector does not have 3 components, or a logical dimension is
        missing.

    See Also
    --------
    volume_integral

    Examples
    --------
    >>> # ½ uᵀ M2n u, as LinearMHD's en_U
    >>> field_energy(u_2form, form=2, weight=n0, domain=out.domain)
    >>> field_energy(E)  # ½ ∫ |E|² dV
    """
    validate_array(data)
    if form not in (None, 0, 1, 2, 3, "v"):
        raise ValueError(f"form must be None, 0, 1, 2, 3 or 'v'; got {form!r}")
    sqrt_g, metric = _geometry(data, domain)
    w = (
        np.ones_like(sqrt_g)
        if weight is None
        else np.asarray(weight, dtype=float) * np.ones_like(sqrt_g)
    )
    w = np.where(np.isfinite(w), w, 0.0)
    vector = "component" in data.dims
    if vector and data.sizes["component"] != 3:
        raise ValueError(
            f"a vector field needs 3 components; got {data.sizes['component']}"
        )
    if not vector and form in (1, 2, "v"):
        raise ValueError(
            f"form={form!r} needs vector components (a 'component' dimension)"
        )
    if vector and form in (0, 3):
        raise ValueError(
            f"form={form} is a scalar; a vector field needs form None, 1, 2 or 'v'"
        )

    if form in (None, 0):
        factor = _spatial_array(w * sqrt_g, data)
        squared = (data**2).sum("component") if vector else data ** 2
    elif form == 3:
        factor = _spatial_array(w / np.where(sqrt_g > 0, sqrt_g, np.inf), data)
        squared = data**2
    else:
        if form == 1:
            tensor = np.linalg.inv(np.moveaxis(metric, (0, 1), (-2, -1)))
            tensor = np.moveaxis(tensor, (-2, -1), (0, 1)) * (w * sqrt_g)
        elif form == 2:
            tensor = metric * (w / np.where(sqrt_g > 0, sqrt_g, np.inf))
        else:
            tensor = metric * (w * sqrt_g)
        spatial = logical_dims(data)
        A = xr.DataArray(
            tensor,
            dims=("component", "component_2", *spatial),
            coords={d: data[d] for d in spatial},
        )
        other = data.rename(component="component_2").drop_vars(
            "component_2", errors="ignore"
        )
        squared = (data.drop_vars("component", errors="ignore") * A * other).sum(
            ("component", "component_2")
        )
        factor = 1.0
    out = normalization * 0.5 * _integrate(squared * factor, quadrature)
    out.attrs = {**_provenance(data), "label": f"energy of {_label(data)}".strip()}
    return out


def gradient(data: xr.DataArray, *, domain=None) -> xr.DataArray:
    """The Cartesian gradient of a scalar field on the logical grid.

    ``∇f = J⁻ᵀ ∂f/∂η``, with the Jacobian ``J`` of the mapping from a struphy ``domain`` (exact)
    or, without one, from the attached ``X``, ``Y``, ``Z`` coordinates. The logical derivatives
    are spectral around periodic directions and second order elsewhere (see
    :func:`plasma_plots.arrays.logical_derivative`). A direction with a single point (a 2-D run)
    is left out: the result is then the gradient within the plane (with the pseudo-inverse of
    the remaining Jacobian columns). For example, ``E = -gradient(phi)``, or the E × B velocity
    ``ẑ × ∇φ`` of a 2-D drift-wave model.

    Parameters
    ----------
    data : xarray.DataArray
        The scalar field, with the dimensions ``eta1``, ``eta2``, ``eta3`` (at least one with
        more than one point) and no ``component`` dimension.
    domain : struphy domain, optional
        The mapping (``out.domain``), for the exact Jacobian. Default: differentiate the ``X``,
        ``Y``, ``Z`` coordinates numerically.

    Returns
    -------
    xarray.DataArray
        The gradient, with a ``component`` dimension ``(x, y, z)`` (coordinates 0, 1, 2) in front
        of the field's own dimensions (e.g. ``t`` is kept), named ``grad_<name>``.

    Raises
    ------
    ValueError
        If a logical dimension is missing, ``data`` has a ``component`` dimension, no direction
        has more than one point, or there is neither a ``domain`` nor ``X``, ``Y``, ``Z``.

    See Also
    --------
    divergence, curl

    Examples
    --------
    >>> E = -gradient(phi)  # Cartesian components (x, y, z), over time
    >>> E = -gradient(phi, domain=out.domain)  # with the exact Jacobian
    """
    from .arrays import logical_derivative, mapping_jacobian, periodicity

    validate_array(data)
    spatial = logical_dims(data)
    missing = [d for d in spatial if d not in data.dims]
    if missing:
        raise ValueError(
            f"gradient needs the logical dimensions {spatial}; {missing} are missing"
        )
    if "component" in data.dims:
        raise ValueError("gradient takes a scalar field; select a component first")
    field = data.transpose(..., *spatial)
    etas = [np.asarray(field[d], dtype=float) for d in spatial]
    varying = [i for i, eta in enumerate(etas) if eta.size > 1]
    if not varying:
        raise ValueError(
            "gradient needs at least one logical direction with more than one point"
        )
    if domain is not None:
        points = np.stack(
            [np.asarray(c, dtype=float) for c in domain(*etas, squeeze_out=False)],
            axis=-1,
        )
        jacobian = np.asarray(domain.jacobian(*etas), dtype=float)
    else:
        if any(name not in field.coords for name in ("X", "Y", "Z")):
            raise ValueError(
                "gradient needs the X, Y, Z coordinates, or a struphy domain"
            )
        points = np.stack(
            [
                np.asarray(field.coords[name].transpose(*spatial), dtype=float)
                for name in ("X", "Y", "Z")
            ],
            axis=-1,
        )
        if len(varying) == 3:
            jacobian = mapping_jacobian(field)
        else:
            jacobian = np.zeros((3, 3, *points.shape[:3]))
            for i in varying:
                kind = periodicity(points, i)
                for a in range(3):
                    jacobian[a, i] = logical_derivative(
                        points[..., a], etas[i], i, kind
                    )
    values = np.asarray(field, dtype=float)
    offset = values.ndim - 3
    derivatives = [
        logical_derivative(values, etas[i], offset + i, periodicity(points, i))
        for i in varying
    ]
    columns = np.moveaxis(jacobian[:, varying], (0, 1), (-2, -1))  # (n1, n2, n3, 3, k)
    transform = np.swapaxes(
        np.linalg.pinv(columns), -1, -2
    )  # pinv(J)^T: (n1, n2, n3, 3, k)
    stacked = np.stack(derivatives, axis=-1)  # (..., n1, n2, n3, k)
    cartesian = np.einsum("...ak,...k->...a", transform, stacked)
    cartesian = np.moveaxis(cartesian, -1, 0)
    out = xr.DataArray(
        cartesian,
        dims=("component", *field.dims),
        coords={**{k: v for k, v in field.coords.items()}, "component": [0, 1, 2]},
        name=f"grad_{field.name}" if field.name else "gradient",
    )
    out.attrs = {**_provenance(data), "label": f"gradient of {_label(data)}".strip()}
    return out


# ---------------------------------------------------------------------------------------------
# Helpers for the struphy-hub examples: errors against exact solutions, single-mode projections,
# vector calculus, local vector components, orbit invariants
# ---------------------------------------------------------------------------------------------


def evaluate_on(data: xr.DataArray, function, args=None) -> xr.DataArray:
    """``function`` evaluated on the coordinates of ``data``, broadcast to its shape.

    E.g. an exact solution ``exact(x, y, z, t)`` on the points of a field, or ``exact(eta1, t)``
    on a 1-D profile.

    Parameters
    ----------
    data : xarray.DataArray
        The array whose coordinates and shape are used.
    function : callable
        Called with the coordinates named by ``args``, positionally, as xarray.DataArrays; returns
        an array (or number) broadcastable to ``data``.
    args : sequence of str, optional
        The coordinates passed positionally, in order. Default: the physical ``X``, ``Y``, ``Z``
        (if attached, else the logical dimensions ``eta1``... that ``data`` has), followed by
        ``t`` if ``data`` has it.

    Returns
    -------
    xarray.DataArray
        The values, broadcast to the shape of ``data`` with its dimensions first.

    Raises
    ------
    ValueError
        If one of ``args`` is not a coordinate of ``data``.

    Examples
    --------
    >>> exact = evaluate_on(u, lambda x, y, z, t: np.sin(x - t))
    """
    if args is None:
        space = (
            ["X", "Y", "Z"]
            if all(c in data.coords for c in ("X", "Y", "Z"))
            else [d for d in logical_dims(data) if d in data.dims]
        )
        args = [*space, *(["t"] if "t" in data.coords else [])]
    missing = [name for name in args if name not in data.coords]
    if missing:
        raise ValueError(
            f"{missing} are not coordinates of {data.name!r}; it has {tuple(data.coords)}"
        )
    values = function(*(data.coords[name] for name in args))
    values = xr.DataArray(values) if not isinstance(values, xr.DataArray) else values
    return values.broadcast_like(data).transpose(*data.dims, ...)


def error(
    data: xr.DataArray,
    exact,
    *,
    norm: str = "rms",
    relative: bool = False,
    dims=None,
    weighted: bool = False,
    domain=None,
    args=None,
) -> xr.DataArray:
    """The error of ``data`` against an exact solution, as a function of the other dimensions.

    Unweighted, the norms are means over the grid points (``l1``, and ``rms`` = ``l2`` its root
    mean square), i.e. integrals over the logical unit cube. With ``weighted`` the norms over the
    logical grid are integrals over the physical volume (with ``|√g|``, from ``domain`` or the
    ``X``, ``Y``, ``Z`` coordinates; see :func:`volume_integral`), and ``rms`` is divided by that
    volume: the proper L2 error on a mapped domain, where a plain mean over grid points is not.

    Parameters
    ----------
    data : xarray.DataArray
        The numerical solution.
    exact : callable, array_like, number or xarray.DataArray
        The exact solution: an array (aligned with ``data``, or broadcast to it) or a function of
        its coordinates, evaluated with :func:`evaluate_on`, e.g. ``lambda x, y, z, t: ...``.
    norm : {"rms", "max", "l1", "l2", "pointwise"}, optional
        ``"pointwise"`` is the difference ``data − exact`` itself; ``"max"`` the largest absolute
        difference; ``"l1"`` the mean (or, weighted, the integral) of ``|data − exact|``;
        ``"l2"`` the square root of the mean (or integral) of ``|data − exact|²``; ``"rms"``
        (default) as ``"l2"``, divided by the volume when weighted.
    relative : bool, optional
        Divide by the same norm of the exact solution; for ``"pointwise"``, by the largest
        ``|exact|`` over the whole array. Default: ``False``.
    dims : str or sequence of str, optional
        The dimensions the norm is taken over. Default: every dimension but ``t``. Weighted
        norms need exactly ``eta1``, ``eta2``, ``eta3``.
    weighted : bool, optional
        Integrate over the physical volume instead of averaging over the grid points (no effect
        on ``"max"`` and ``"pointwise"``). Default: ``False``.
    domain : struphy domain, optional
        The mapping (``out.domain``), for the exact ``|√g|`` of weighted norms. Default: from
        the ``X``, ``Y``, ``Z`` coordinates.
    args : sequence of str, optional
        The coordinates passed to a callable ``exact``, as in :func:`evaluate_on`. Default: ``X``,
        ``Y``, ``Z`` (or the logical dimensions), then ``t``.

    Returns
    -------
    xarray.DataArray
        The error as a function of the dimensions not in ``dims`` (typically ``t``); for
        ``"pointwise"``, an array like ``data``. Labeled e.g. ``"relative rms error of ..."``.

    Raises
    ------
    ValueError
        If ``norm`` is unknown, or a weighted norm is not over ``eta1``, ``eta2``, ``eta3``.

    See Also
    --------
    convergence_order : The order of a series of errors.

    Examples
    --------
    >>> error(T, exact, relative=True)  # relative RMS error over time
    >>> error(T, exact, norm="max")  # largest pointwise error
    >>> # √∫|u − u_exact|² dV
    >>> error(u, lambda x, y, z, t: np.sin(x - t), norm="l2", weighted=True)
    """
    validate_array(data)
    if callable(exact):
        reference = evaluate_on(data, exact, args)
    elif isinstance(exact, xr.DataArray):
        reference = exact.broadcast_like(data)
    else:  # a plain array (or number) on the grid of data
        exact = np.asarray(exact)
        reference = (
            data.copy(data=np.broadcast_to(exact, data.shape))
            if exact.ndim
            else xr.full_like(data, float(exact))
        )
    difference = data - reference
    if norm == "pointwise":
        out = difference / abs(reference).max() if relative else difference
        out.attrs = {**_provenance(data), "label": f"error of {_label(data)}".strip()}
        return out
    if norm not in ("max", "rms", "l1", "l2"):
        raise ValueError(
            f"norm must be 'pointwise', 'max', 'rms', 'l1' or 'l2'; got {norm!r}"
        )
    dims = (
        [d for d in data.dims if d != "t"]
        if dims is None
        else ([dims] if isinstance(dims, str) else list(dims))
    )

    def measure(values):
        # unweighted: averages over the grid points, i.e. integrals over the logical unit cube;
        # weighted: integrals over the physical volume (rms divides by that volume)
        if norm == "max":
            return abs(values).max(dims)
        power = 1 if norm == "l1" else 2
        if weighted:
            spatial = logical_dims(values)
            if not set(spatial) <= set(values.dims) or set(dims) != set(spatial):
                raise ValueError(
                    f"weighted norms integrate over {spatial} (the default dims)"
                )
            integral = volume_integral(abs(values) ** power, domain=domain)
            if norm == "rms":
                ones = xr.ones_like(
                    values.isel(
                        {d: 0 for d in values.dims if d not in spatial}, drop=True
                    )
                )
                integral = integral / volume_integral(ones, domain=domain)
        else:
            integral = (abs(values) ** power).mean(dims)
        return integral if norm == "l1" else np.sqrt(integral)

    out = measure(difference)
    if relative:
        out = out / measure(reference)
    out.attrs = {
        **_provenance(data),
        "label": f"{'relative ' if relative else ''}{norm} error of {_label(data)}".strip(),
    }
    return out


def project_mode(
    data: xr.DataArray,
    *,
    dim: str,
    number: float,
    kind: str = "sin",
    period: float = 1.0,
    bin_correction: bool = False,
) -> xr.DataArray:
    """The amplitude of one Fourier mode along a periodic dimension, as a function of the rest.

    The amplitude is twice the mean over the samples of ``data`` times the basis function, e.g.
    ``2 ⟨f sin(…)⟩``.

    Parameters
    ----------
    data : xarray.DataArray
        The field. ``dim`` must sample one full period uniformly (a duplicate endpoint is
        dropped).
    dim : str
        The periodic dimension, e.g. ``"eta1"``.
    number : float
        The mode number, not a wavenumber: ``k = 2π number / L``.
    kind : {"sin", "cos", "complex"}, optional
        ``"sin"`` (default) gives ``a`` of ``a sin(2π number x / period)``: ``2 ⟨f sin(…)⟩``;
        ``"cos"`` the cosine amplitude, and ``"complex"`` the complex amplitude
        ``2 ⟨f exp(−i…)⟩`` (``abs()`` the amplitude, ``np.angle()`` the phase).
    period : float, optional
        The period of ``dim``. Default: 1, the logical unit interval.
    bin_correction : bool, optional
        Undo the damping of binned particle data, whose bins average the mode over their width
        ``h``: divide by ``sinc(number h / period)``. Default: ``False``.

    Returns
    -------
    xarray.DataArray
        The amplitude as a function of the other dimensions (complex for ``"complex"``), named
        e.g. ``sin_1``.

    Raises
    ------
    ValueError
        If ``kind`` is unknown, or ``dim`` does not sample one full period uniformly.

    Examples
    --------
    >>> amplitude = project_mode(e1, dim="eta1", number=1)  # sine amplitude over t
    >>> phase = np.angle(project_mode(rho, dim="eta2", number=3, kind="complex"))
    """
    from .spectral import drop_periodic_endpoint

    if kind not in ("sin", "cos", "complex"):
        raise ValueError(f"kind must be 'sin', 'cos' or 'complex'; got {kind!r}")
    trimmed = drop_periodic_endpoint(data, dim, period=period)
    x = np.asarray(trimmed[dim], dtype=float)
    h = x[1] - x[0] if x.size > 1 else period
    if (
        x.size < 2
        or not np.allclose(np.diff(x), h)
        or not np.isclose(x.size * h, period, rtol=1e-6)
    ):
        raise ValueError(f"{dim!r} must sample one full period ({period}) uniformly")
    phase = 2 * np.pi * number * trimmed[dim] / period
    basis = {
        "sin": np.sin(phase),
        "cos": np.cos(phase),
        "complex": np.exp(-1j * phase),
    }[kind]
    amplitude = 2 * (trimmed * basis).mean(dim)
    if bin_correction:
        amplitude = amplitude / np.sinc(number * h / period)
    amplitude.name = f"{kind}_{number:g}"
    amplitude.attrs = {
        **_provenance(data),
        "label": f"{kind} amplitude of mode {number:g} along {dim}",
    }
    return amplitude


def _cartesian_vector(vector: xr.DataArray, components: str) -> xr.DataArray:
    if "component" not in vector.dims or vector.sizes["component"] != 3:
        raise ValueError(
            "expected a vector field with a 'component' dimension of size 3"
        )
    if components == "contravariant":
        from .pyvista_plots import push_forward

        return push_forward(vector)
    if components != "cartesian":
        raise ValueError(
            f'components must be "cartesian" or "contravariant"; got {components!r}'
        )
    return vector


def _jacobian_of_components(vector, domain):
    """``d v_a / d x_b`` as an array over (a, b, ...)."""
    rows = [
        gradient(vector.isel(component=a, drop=True), domain=domain) for a in range(3)
    ]
    return xr.concat(rows, dim="row")  # (row a, component b, ...)


def divergence(
    vector: xr.DataArray, *, components: str = "cartesian", domain=None
) -> xr.DataArray:
    """The divergence ``sum_a d v_a / d x_a`` of a vector field on a mapped domain.

    Derivatives as in :func:`gradient`: spectral around periodic angles; a flat direction (2-D
    run) is left out. E.g. a ``div B`` check of an MHD run.

    Parameters
    ----------
    vector : xarray.DataArray
        The vector field, with a ``component`` dimension of size 3 and the dimensions ``eta1``,
        ``eta2``, ``eta3``.
    components : {"cartesian", "contravariant"}, optional
        What the components are, as for the 3-D views: ``"cartesian"`` (default) ``(x, y, z)``,
        or ``"contravariant"`` components, which are pushed forward first.
    domain : struphy domain, optional
        The mapping (``out.domain``), for the exact Jacobian. Default: from the ``X``, ``Y``,
        ``Z`` coordinates.

    Returns
    -------
    xarray.DataArray
        The divergence, over the field's dimensions without ``component``, named
        ``div_<name>``.

    Raises
    ------
    ValueError
        If ``vector`` has no ``component`` dimension of size 3, ``components`` is unknown, or as
        for :func:`gradient`.

    See Also
    --------
    gradient, curl

    Examples
    --------
    >>> div_B = divergence(B)  # should stay at round-off
    >>> norm(div_B).plasma.plot.timeseries()
    """
    cartesian = _cartesian_vector(vector, components)
    jac = _jacobian_of_components(cartesian, domain)
    out = sum(jac.isel(row=a, component=a, drop=True) for a in range(3))
    out.name = f"div_{vector.name}" if vector.name else "divergence"
    out.attrs = {
        **_provenance(vector),
        "label": f"divergence of {_label(vector)}".strip(),
    }
    return out


def curl(
    vector: xr.DataArray, *, components: str = "cartesian", domain=None
) -> xr.DataArray:
    """The curl of a vector field on a mapped domain, in Cartesian components ``(x, y, z)``.

    Derivatives as for :func:`divergence`. E.g. the current ``J = ∇ × B``, or the vorticity of a
    flow; for a 2-D field in the ``x``-``y`` plane only the ``z`` component is non-zero.

    Parameters
    ----------
    vector : xarray.DataArray
        The vector field, with a ``component`` dimension of size 3 and the dimensions ``eta1``,
        ``eta2``, ``eta3``.
    components : {"cartesian", "contravariant"}, optional
        What the components are: ``"cartesian"`` (default) ``(x, y, z)``, or ``"contravariant"``
        components, which are pushed forward first.
    domain : struphy domain, optional
        The mapping (``out.domain``), for the exact Jacobian. Default: from the ``X``, ``Y``,
        ``Z`` coordinates.

    Returns
    -------
    xarray.DataArray
        The curl, with a ``component`` dimension ``(x, y, z)`` (coordinates 0, 1, 2) first,
        named ``curl_<name>``.

    Raises
    ------
    ValueError
        As for :func:`divergence`.

    See Also
    --------
    gradient, divergence

    Examples
    --------
    >>> J = curl(B)  # the current, in Cartesian components
    >>> vorticity = curl(u).sel(component=2)
    """
    cartesian = _cartesian_vector(vector, components)
    jac = _jacobian_of_components(cartesian, domain)

    def d(a, b):
        return jac.isel(row=a, component=b, drop=True)

    out = xr.concat(
        [d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)], dim="component"
    )
    out = out.assign_coords(component=[0, 1, 2]).transpose("component", ...)
    out.name = f"curl_{vector.name}" if vector.name else "curl"
    out.attrs = {**_provenance(vector), "label": f"curl of {_label(vector)}".strip()}
    return out


def flux_function(vector: xr.DataArray) -> xr.DataArray:
    """The flux function (or stream function) ``A`` of a 2-D, divergence-free in-plane field.

    For a magnetic field ``B_x = ∂A/∂y``, ``B_y = −∂A/∂x`` (so ``B = ∇A × ẑ``); its contour
    lines are the field lines, e.g. drawn over the current with a slice's ``contours_of``. For a
    velocity it is the stream function. ``A`` is integrated along the grid lines (trapezoidal
    rule), ``A(x, y) = ∫ B_x(x₀, y′) dy′ − ∫ B_y(x′, y) dx′``, and shifted to zero mean.

    Parameters
    ----------
    vector : xarray.DataArray
        The field in Cartesian components, with a ``component`` dimension of size 3 (only
        ``x`` and ``y`` are used). It must lie on a Cartesian grid in the ``x``-``y`` plane:
        exactly two logical directions with more than one point, physical ``X`` depending on
        one of them and ``Y`` on the other, e.g. a slab or periodic box.

    Returns
    -------
    xarray.DataArray
        ``A``, named ``flux_function``, as a function of every dimension of ``vector`` except
        ``component``.

    Raises
    ------
    ValueError
        If ``vector`` is not a 3-component field on such a grid.

    Examples
    --------
    >>> A = flux_function(B.isel(t=-1))
    >>> J = curl(B).isel(t=-1, component=2)
    >>> J.plasma.plot.slice(overlays={"contours_of": A})
    """
    cartesian = _cartesian_vector(vector, "cartesian").isel(component=[0, 1])
    spatial = logical_dims(cartesian)
    field = cartesian.squeeze(
        [d for d in spatial if d in cartesian.dims and cartesian.sizes[d] == 1],
        drop=False,
    )
    grid = [d for d in spatial if d in field.dims and field.sizes[d] > 1]
    if len(grid) != 2 or not all(c in field.coords for c in ("X", "Y")):
        raise ValueError(
            "flux_function needs a 2-D field on two logical directions with X, Y coordinates"
        )
    X = np.asarray(field.coords["X"].squeeze(drop=True).transpose(*grid), dtype=float)
    Y = np.asarray(field.coords["Y"].squeeze(drop=True).transpose(*grid), dtype=float)
    if np.allclose(X, X[:, :1]) and np.allclose(Y, Y[:1, :]):
        xdim, ydim = grid
    elif np.allclose(X, X[:1, :]) and np.allclose(Y, Y[:, :1]):
        ydim, xdim = grid
    else:
        raise ValueError(
            "flux_function needs a Cartesian grid: X along one logical direction and Y along the other"
        )
    x = np.asarray(field.coords["X"].isel({ydim: 0}).squeeze(drop=True), dtype=float)
    y = np.asarray(field.coords["Y"].isel({xdim: 0}).squeeze(drop=True), dtype=float)
    bx = field.isel(component=0, drop=True).transpose(..., xdim, ydim)
    by = field.isel(component=1, drop=True).transpose(..., xdim, ydim)

    def cumulative(values, coordinate, axis):
        steps = np.diff(coordinate)
        shape = [1] * values.ndim
        shape[axis] = steps.size
        pairs = 0.5 * (
            np.take(values, range(1, values.shape[axis]), axis=axis)
            + np.take(values, range(values.shape[axis] - 1), axis=axis)
        )
        integral = np.cumsum(pairs * steps.reshape(shape), axis=axis)
        return np.concatenate(
            [np.zeros_like(np.take(values, [0], axis=axis)), integral], axis=axis
        )

    bx_values, by_values = np.asarray(bx, dtype=float), np.asarray(by, dtype=float)
    # A(x, y) = int_0^y B_x(x0, y') dy' - int_0^x B_y(x', y) dx'
    along_y = cumulative(np.take(bx_values, [0], axis=-2), y, -1)
    along_x = cumulative(by_values, x, -2)
    values = along_y - along_x
    values = values - values.mean(axis=(-2, -1), keepdims=True)
    out = bx.copy(data=values)
    out = out.transpose(*[d for d in vector.dims if d != "component" and d in out.dims])
    out.name = "flux_function"
    out.attrs = {
        **_provenance(vector),
        "label": f"flux function of {_label(vector)}".strip(),
    }
    return out


def _angles(data: xr.DataArray, R0: float = 0.0, Z0: float = 0.0):
    X, Y, Z = (data.coords[n] for n in ("X", "Y", "Z"))
    R = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    theta = np.arctan2(Z - Z0, R - R0)
    return R, phi, theta


def cylindrical_components(vector: xr.DataArray) -> xr.DataArray:
    """Cartesian components rotated to cylindrical ones ``(R, phi, Z)`` about the ``Z`` axis.

    The rotation is by the toroidal angle ``φ = atan2(Y, X)`` at every point of the field.

    Parameters
    ----------
    vector : xarray.DataArray
        The field in Cartesian components, with a ``component`` dimension of size 3 and the
        ``X``, ``Y``, ``Z`` coordinates.

    Returns
    -------
    xarray.DataArray
        The components ``v_R``, ``v_φ``, ``v_Z``, with ``component`` coordinates ``"R"``,
        ``"phi"``, ``"Z"`` and the dimensions of ``vector``.

    Raises
    ------
    ValueError
        If ``vector`` has no ``component`` dimension of size 3.

    See Also
    --------
    toroidal_components

    Examples
    --------
    >>> E_R = cylindrical_components(-gradient(phi)).sel(component="R")
    """
    cartesian = _cartesian_vector(vector, "cartesian")
    _, phi, _ = _angles(cartesian)
    vx, vy, vz = (cartesian.isel(component=i, drop=True) for i in range(3))
    out = (
        xr.concat(
            [
                vx * np.cos(phi) + vy * np.sin(phi),
                -vx * np.sin(phi) + vy * np.cos(phi),
                vz,
            ],
            dim="component",
        )
        .assign_coords(component=["R", "phi", "Z"])
        .transpose(*vector.dims)
    )
    out.name, out.attrs = (
        vector.name,
        {
            **vector.attrs,
            "label": f"{_label(vector)} (R, phi, Z)",
        },
    )
    return out


def toroidal_components(
    vector: xr.DataArray, *, R0: float, Z0: float = 0.0
) -> xr.DataArray:
    """Cartesian components rotated to the local ``(radial, poloidal, toroidal)`` directions.

    The directions are those about a circular magnetic axis at major radius ``R0`` (height
    ``Z0``), at every point. The radial direction points away from the axis in the poloidal
    plane, the poloidal one along ``θ = atan2(Z − Z0, R − R0)``, the toroidal one along ``φ``:
    the natural components for waves in a tokamak or torus.

    Parameters
    ----------
    vector : xarray.DataArray
        The field in Cartesian components, with a ``component`` dimension of size 3 and the
        ``X``, ``Y``, ``Z`` coordinates.
    R0 : float
        The major radius of the magnetic axis.
    Z0 : float, optional
        The height of the magnetic axis. Default: 0.

    Returns
    -------
    xarray.DataArray
        The components, with ``component`` coordinates ``"radial"``, ``"poloidal"``,
        ``"toroidal"`` and the dimensions of ``vector``.

    Raises
    ------
    ValueError
        If ``vector`` has no ``component`` dimension of size 3.

    See Also
    --------
    cylindrical_components

    Examples
    --------
    >>> local = toroidal_components(u, R0=3.0)
    >>> local.sel(component="poloidal").isel(t=-1, eta3=0).plasma.plot.slice()
    """
    cartesian = _cartesian_vector(vector, "cartesian")
    _, phi, theta = _angles(cartesian, R0, Z0)
    vx, vy, vz = (cartesian.isel(component=i, drop=True) for i in range(3))
    v_R = vx * np.cos(phi) + vy * np.sin(phi)
    v_phi = -vx * np.sin(phi) + vy * np.cos(phi)
    out = (
        xr.concat(
            [
                v_R * np.cos(theta) + vz * np.sin(theta),
                -v_R * np.sin(theta) + vz * np.cos(theta),
                v_phi,
            ],
            dim="component",
        )
        .assign_coords(component=["radial", "poloidal", "toroidal"])
        .transpose(*vector.dims)
    )
    out.name, out.attrs = (
        vector.name,
        {
            **vector.attrs,
            "label": f"{_label(vector)} (radial, poloidal, toroidal)",
        },
    )
    return out


def polar_coordinates(data: xr.DataArray, *, center=(0.0, 0.0)) -> xr.DataArray:
    """Attach the polar coordinates ``r`` and ``theta`` of each point in the ``X``-``Y`` plane.

    E.g. for profiles against the radius.

    Parameters
    ----------
    data : xarray.DataArray
        The array, with the ``X`` and ``Y`` coordinates.
    center : (float, float), optional
        The origin ``(X, Y)`` of the polar coordinates. Default: ``(0.0, 0.0)``.

    Returns
    -------
    xarray.DataArray
        ``data`` with the extra coordinates ``r`` and ``theta`` (radians, in ``(−π, π]``) of its
        points about ``center``.

    Examples
    --------
    >>> n_polar = polar_coordinates(n, center=(0.0, 0.0)).isel(t=-1, eta3=0)
    """
    X, Y = data.coords["X"], data.coords["Y"]
    return data.assign_coords(
        r=np.hypot(X - center[0], Y - center[1]),
        theta=np.arctan2(Y - center[1], X - center[0]),
    )


def orbit_invariants(orbits: xr.Dataset, *, absB=None) -> xr.Dataset:
    """Kinetic invariants of saved marker orbits, over ``(t, marker)``.

    Computes whatever the saved quantities allow:

    * ``speed`` ``|v|`` from ``v1``, ``v2``, ``v3`` (full orbits);
    * with ``v_par``, ``mu``, the positions ``x``, ``y``, ``z`` and ``absB``: the guiding-centre
      ``energy`` ``v_par² / 2 + mu |B|`` and the ``pitch`` ``v_par / v``, with
      ``v = √(2 energy)``.

    Their drift, e.g. :func:`relative_error` of the energy, measures the pusher's accuracy.

    Parameters
    ----------
    orbits : xarray.Dataset
        The orbits product, with variables over ``(t, marker)``.
    absB : callable, optional
        ``|B|(x, y, z)`` as a function of the physical positions (numpy arrays), e.g.
        ``lambda x, y, z: out.equil.absB0(*out.domain.inverse_map(x, y, z))``. Needed for the
        energy and the pitch.

    Returns
    -------
    xarray.Dataset
        The invariants that could be computed (``speed``, ``energy``, ``pitch``), over
        ``(t, marker)``. Samples where a marker has left the domain are NaN.

    Raises
    ------
    ValueError
        If no invariant can be computed (need ``v1``..``v3``, or ``v_par``, ``mu`` and ``absB``).

    See Also
    --------
    classify_orbits, bounce_period

    Examples
    --------
    >>> invariants = orbit_invariants(orbits, absB=absB)
    >>> relative_error(invariants.energy.isel(marker=0)).plasma.plot.timeseries()
    """
    from .plotting import _alive

    alive = xr.DataArray(
        _alive(orbits.transpose("t", "marker", ...)), dims=("t", "marker")
    )
    out = {}
    if all(n in orbits for n in ("v1", "v2", "v3")):
        out["speed"] = np.sqrt(orbits.v1**2 + orbits.v2**2 + orbits.v3**2)
    if absB is not None and all(n in orbits for n in ("v_par", "mu", "x", "y", "z")):
        B = xr.DataArray(
            np.asarray(
                absB(np.asarray(orbits.x), np.asarray(orbits.y), np.asarray(orbits.z)),
                dtype=float,
            ),
            dims=orbits.x.dims,
            coords=orbits.x.coords,
        )
        energy = 0.5 * orbits.v_par**2 + orbits.mu * B
        out["energy"] = energy
        out["pitch"] = orbits.v_par / np.sqrt(2 * energy)
    if not out:
        raise ValueError(
            "no invariant can be computed from these orbits (need v1..v3, or v_par, mu and absB)"
        )
    return xr.Dataset(
        {name: values.where(alive) for name, values in out.items()},
        attrs=_provenance(orbits),
    )


def bounce_period(orbits: xr.Dataset, *, v_par: str = "v_par") -> xr.DataArray:
    """The bounce period of each trapped marker.

    Twice the mean time between reversals of its parallel velocity; the reversal times are
    interpolated linearly between the saved samples.

    Parameters
    ----------
    orbits : xarray.Dataset
        The orbits product, with variables over ``(t, marker)``.
    v_par : str, optional
        The name of the parallel-velocity variable. Default: ``"v_par"``.

    Returns
    -------
    xarray.DataArray
        The period over ``marker``, named ``bounce_period``; NaN for markers with fewer than two
        reversals, e.g. passing ones.

    See Also
    --------
    classify_orbits

    Examples
    --------
    >>> periods = bounce_period(orbits)
    >>> periods.where(classify_orbits(orbits) == 1).mean()
    """
    velocity = orbits[v_par].transpose("t", "marker")
    t = np.asarray(orbits.t, dtype=float)
    periods = []
    for m in range(velocity.sizes["marker"]):
        v = np.asarray(velocity.isel(marker=m))
        sign = np.sign(v)
        flips = np.flatnonzero((sign[1:] * sign[:-1]) < 0)
        # linear interpolation of the crossing times
        crossings = t[flips] - v[flips] * (t[flips + 1] - t[flips]) / (
            v[flips + 1] - v[flips]
        )
        periods.append(
            2 * np.mean(np.diff(crossings)) if crossings.size >= 2 else np.nan
        )
    out = xr.DataArray(
        np.asarray(periods),
        dims="marker",
        coords={"marker": orbits.marker},
        name="bounce_period",
    )
    out.attrs = {**_provenance(orbits), "label": "bounce period"}
    return out
