"""Numerical diagnostics returning values and labeled arrays, without rendering."""

from dataclasses import dataclass

import numpy as np
import xarray as xr

from .arrays import validate_array


def _label(data):
    return data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""


@dataclass(frozen=True)
class GrowthFit:
    """Configuration for an exponential growth-rate fit."""

    window: tuple[float | None, float | None] = (None, None)
    amplitude_from_quadratic: bool = False


@dataclass(frozen=True)
class FitResult:
    rate: float
    intercept: float
    time: np.ndarray
    fitted: np.ndarray


@dataclass(frozen=True)
class ConvergenceFit:
    order: float
    constant: float
    sizes: np.ndarray
    fitted: np.ndarray


def convergence_order(sizes, errors) -> ConvergenceFit | None:
    """Fit ``error = constant * size**order`` in log-log space.

    ``sizes`` is typically a resolution (points per cell, coarser to finer) or a step size
    (``dt``); ``errors`` are the corresponding, necessarily positive, error norms. ``order`` is
    negative when the error shrinks as ``sizes`` grows (e.g. more points per cell), and positive
    when it shrinks as ``sizes`` shrinks (e.g. a smaller ``dt``). Returns ``None`` with fewer than
    two valid (finite, positive) samples.
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
    """Fit ``exp(rate*t + intercept)`` using only finite, positive samples."""
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
    signal = np.log(np.sqrt(values[valid])) if fit.amplitude_from_quadratic else np.log(values[valid])
    rate, intercept = np.polyfit(selected_time, signal, 1)
    scale = 2.0 if fit.amplitude_from_quadratic else 1.0
    fitted = np.exp(scale * (rate * selected_time + intercept))
    return FitResult(float(rate), float(intercept), selected_time, fitted)


def envelope(data: xr.DataArray) -> xr.DataArray:
    """Local maxima of a time series: the interior samples not smaller than their neighbours."""
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
    the raw series would fit the oscillation. ``fit.window`` restricts the peaks that are used.
    The rate is negative for damping.
    """
    return growth_rate(envelope(data), fit)


def norm(data: xr.DataArray, *, dims=None, squared: bool = False) -> xr.DataArray:
    """L2 norm over ``dims`` (default: every dimension except ``t``), as a function of the rest."""
    validate_array(data)
    dims = [dim for dim in data.dims if dim != "t"] if dims is None else list(dims)
    total = (data**2).sum(dims)
    out = total if squared else np.sqrt(total)
    out.attrs = {key: value for key, value in data.attrs.items() if key in ("run", "run_name")}
    label = _label(data)
    out.attrs["label"] = f"squared norm of {label}".strip() if squared else f"norm of {label}".strip()
    return out


def drift(data: xr.DataArray, *, ref=None) -> xr.DataArray:
    """Signed deviation from an explicit reference or the first time sample."""
    validate_array(data, required_dims=("t",))
    reference = data.isel(t=0) if ref is None else ref
    out = data - reference
    out.attrs = dict(data.attrs)
    out.attrs["label"] = f"{_label(data)} drift".strip()
    return out


SPATIAL_DIMS = ("eta1", "eta2", "eta3")
VELOCITY_DIMS = ("v1", "v2", "v3")


def _provenance(data: xr.DataArray) -> dict:
    return {key: value for key, value in data.attrs.items() if key in ("run", "run_name")}


def _select_dims(data: xr.DataArray, dims, default) -> list[str]:
    if dims is None:
        selected = [dim for dim in default if dim in data.dims]
        if not selected:
            raise ValueError(f"{data.name!r} has none of the dimensions {default}; its dimensions are {data.dims}")
        return selected
    selected = [dims] if isinstance(dims, str) else list(dims)
    missing = [dim for dim in selected if dim not in data.dims]
    if missing:
        raise ValueError(f"{data.name!r} has no dimensions {missing}; its dimensions are {data.dims}")
    return selected


def spatial_average(data: xr.DataArray, *, dims=None) -> xr.DataArray:
    """Mean over the logical space dimensions, e.g. a binned f(t, eta1, v1) becomes f(t, v1).

    ``dims`` defaults to every one of ``eta1``, ``eta2``, ``eta3`` that ``data`` has. The mean is
    uniform in the logical coordinates, which is the volume average on a Cartesian domain; on a
    mapped domain it is not weighted by the Jacobian. Physical ``X``, ``Y``, ``Z`` coordinates
    that depend on the averaged dimensions are dropped.
    """
    validate_array(data)
    averaged = _select_dims(data, dims, SPATIAL_DIMS)
    out = data.mean(averaged, keep_attrs=True)
    out.attrs["label"] = f"average of {_label(data)}".strip()
    out.attrs.pop("long_name", None)
    return out


def _bin_widths(data: xr.DataArray, dim: str) -> xr.DataArray:
    coordinate = np.asarray(data.coords[dim]) if dim in data.coords else None
    if coordinate is None or len(coordinate) < 2:
        raise ValueError(f"dimension {dim!r} needs a coordinate with at least two bins")
    return xr.DataArray(np.gradient(coordinate), dims=(dim,), coords={dim: data.coords[dim]})


def velocity_moments(f: xr.DataArray, *, dims=None) -> xr.Dataset:
    """Moments of a binned distribution function over its velocity dimensions.

    ``dims`` defaults to every one of ``v1``, ``v2``, ``v3`` that ``f`` has; the moments are
    functions of the remaining dimensions, for example ``(t, eta1)`` for an ``e1_v1`` product.
    The integrals are sums over the bins, weighted by the bin widths.

    Returns a Dataset with

    * ``density``: the zeroth moment, :math:`\\int f\\,\\mathrm{d}v`.
    * ``mean_<dim>``: the mean velocity :math:`u = \\int v f\\,\\mathrm{d}v / n` along each dimension.
    * ``variance_<dim>``: :math:`\\int (v-u)^2 f\\,\\mathrm{d}v / n`. In normalized units this is the
      temperature over the particle mass along that direction, :math:`T/m`.

    Where the density is not positive, the mean and variance are NaN. A ``delta_f`` product has
    only the density, which is then the density perturbation, because its mean and variance are
    not defined. The values keep the normalization of the run; see ``Output.to_si``.
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
    """Absolute relative deviation from an explicit reference or first sample."""
    validate_array(data, required_dims=("t",))
    reference = data.isel(t=0) if ref is None else ref
    if np.any(np.asarray(reference) == 0):
        raise ValueError("cannot take a relative error against a reference of zero")
    out = abs(data - reference) / abs(reference)
    out.attrs = {key: value for key, value in data.attrs.items() if key in ("run", "run_name")}
    out.attrs.update(label=f"relative error of {_label(data)}".strip(), units="")
    return out.isel(t=slice(1, None)) if skip_first else out


ORBIT_CLASSES = {0: "passing", 1: "trapped", -1: "lost"}


def classify_orbits(orbits: xr.Dataset, *, v_par: str = "v_par") -> xr.DataArray:
    """Classify each marker of an orbits product as passing (0), trapped (1) or lost (-1).

    The same criteria as Struphy's ``post_process_orbit_classification``: a marker is trapped if
    its parallel velocity ``v_par`` ever has the opposite sign to its initial one, and lost if at
    any saved time every quantity is zero (how Struphy stores a marker that has left the domain).
    Lost takes precedence over trapped. Returns a ``(marker,)`` array of integer codes; the
    names are in ``attrs["flag_meanings"]`` and in :data:`ORBIT_CLASSES`.
    """
    if v_par not in orbits.data_vars:
        raise ValueError(
            f"orbit classification needs the parallel velocity {v_par!r}; this dataset has {tuple(orbits.data_vars)}"
        )
    if set(orbits[v_par].dims) != {"t", "marker"}:
        raise ValueError(f"{v_par!r} must have dims ('t', 'marker'); got {orbits[v_par].dims}")
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
        **{key: value for key, value in orbits.attrs.items() if key in ("run", "run_name")},
        "label": "orbit classification",
        "flag_values": list(ORBIT_CLASSES),
        "flag_meanings": " ".join(ORBIT_CLASSES.values()),
    }
    return codes


@dataclass(frozen=True)
class BranchFit:
    """A dispersion branch fitted as ``omega = velocity * k``."""

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
    """Fit ``omega = v * k`` to each of ``n_branches`` straight ridges in a (omega, k) power
    spectrum (as returned by :func:`power_spectrum`), with no theoretical curve to guide the
    search -- useful when several linear wave branches are excited at once and there is nothing to
    search around yet. (For a single, possibly non-linear branch with a known theoretical curve to
    guide the search instead, take the frequency of maximum power in a window of ``spectrum``
    around that curve at each k, rather than this blind approach.)

    Only non-negative ``omega`` and ``k`` are scanned (a real signal's spectrum is symmetric under
    ``(k, omega) -> (-k, -omega)``, so every branch already appears on both sides of ``k = 0``). At
    each remaining k, the local maxima of the spectrum along omega are found (see
    :func:`_local_maxima`); a k column contributes to the fit only where it has exactly
    ``n_branches`` maxima above ``noise_level`` times that column's own peak power, taken in
    increasing-omega order. ``k_range`` restricts which non-negative k's are scanned (default:
    ``(k.max() / 8, k.max() / 2)``, which in practice skips both the low-k region where branches
    have not yet separated, and the folded Nyquist edge).

    Returns one :class:`BranchFit` per branch, in increasing-omega order at the low-k end of
    ``k_range``; ``.velocity`` is the fitted slope, ``.k``/``.omega`` the ridge points used.
    """
    if not {"omega", "k"} <= set(spectrum.dims):
        raise ValueError(f"spectrum must have dims 'omega' and 'k'; got {spectrum.dims}")
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


def power_spectrum(data: xr.DataArray, *, dim: str | None = None, detrend: bool = True) -> xr.DataArray:
    """The 2-D power spectrum of a ``(t, dim)`` signal: a plain space-time FFT, as a function of
    angular frequency and wavenumber -- the basis of a dispersion-relation plot
    (:meth:`~struphy_plots.accessors.ArrayPlots.dispersion`), independent of Struphy.

    ``dim`` defaults to the sole dimension other than ``t``; the array must have exactly these
    two dimensions, each on a uniform grid. ``detrend`` removes the time-mean at each point of
    ``dim`` first, which otherwise dominates the spectrum as a spurious zero-frequency line.
    Built on :func:`struphy_plots.spectral.fft`, so it shares its conventions: coefficients are
    divided by the sample counts, and the power sums to the mean square of the signal.
    """
    validate_array(data, required_dims=("t",))
    others = [d for d in data.dims if d != "t"]
    if dim is None:
        if len(others) != 1:
            raise ValueError(f"dim is required unless data has exactly one dimension besides 't'; got {data.dims}")
        dim = others[0]
    elif dim not in data.dims:
        raise ValueError(f"{dim!r} is not a dimension of this array; its dimensions are {data.dims}")
    if set(data.dims) != {"t", dim}:
        raise ValueError(f"select every dimension except 't' and {dim!r} first; got {data.dims}")
    from .spectral import fft

    signal = data.transpose("t", dim).reset_coords(drop=True)
    coefficients = fft(fft(signal, dim="t", detrend=detrend), dim=dim)
    spectrum = (abs(coefficients) ** 2).rename({f"k_{dim}": "k"}).transpose("omega", "k")
    spectrum.name = "power"
    spectrum.attrs = {"label": f"power spectrum of {_label(data)}".strip(), "units": ""}
    return spectrum


# ---------------------------------------------------------------------------------------------
# Volume integrals and field energies (after struphy's TAE_example_Shrut field_energies.py)
# ---------------------------------------------------------------------------------------------


def quadrature_weights(coordinate) -> np.ndarray:
    """Quadrature weights for samples of a logical coordinate in ``[0, 1]``.

    Struphy's cell centers (uniform, half a cell from each end) get the midpoint rule, which
    integrates over the whole unit interval; any other grid gets the trapezoidal rule over the
    sampled range. A single point (a 2-D run's flat direction) has weight 1.
    """
    x = np.asarray(coordinate, dtype=float)
    if x.size == 1:
        return np.ones(1)
    h = np.diff(x)
    if np.allclose(h, h[0]) and np.isclose(x[0], h[0] / 2) and np.isclose(x[-1], 1 - h[0] / 2):
        return np.full(x.size, h[0])
    weights = np.empty(x.size)
    weights[1:-1] = (x[2:] - x[:-2]) / 2
    weights[0], weights[-1] = h[0] / 2, h[-1] / 2
    return weights


def _geometry(data: xr.DataArray, domain=None):
    """``|sqrt g|`` and the metric ``G`` (3, 3, n1, n2, n3) on the logical grid of ``data``: from a
    struphy ``domain`` (exact), or from the Jacobian of the attached X, Y, Z coordinates.
    """
    missing = [d for d in SPATIAL_DIMS if d not in data.dims]
    if missing:
        raise ValueError(f"integrals need the logical dimensions eta1, eta2, eta3; {missing} are missing")
    if domain is not None:
        etas = [np.asarray(data[d], dtype=float) for d in SPATIAL_DIMS]
        sqrt_g = np.abs(np.asarray(domain.jacobian_det(*etas), dtype=float))
        metric = np.asarray(domain.metric(*etas), dtype=float)
        return sqrt_g.reshape([data.sizes[d] for d in SPATIAL_DIMS]), metric
    from .arrays import mapping_jacobian

    jacobian = mapping_jacobian(data)
    sqrt_g = np.abs(np.linalg.det(np.moveaxis(jacobian, (0, 1), (-2, -1))))
    metric = np.einsum("ai...,aj...->ij...", jacobian, jacobian)
    return sqrt_g, metric


def _integrate(integrand: xr.DataArray, quadrature=None) -> xr.DataArray:
    weights = 1.0
    for dim in SPATIAL_DIMS:
        given = (quadrature or {}).get(dim)
        values = quadrature_weights(integrand[dim]) if given is None else np.asarray(given, dtype=float)
        if values.size != integrand.sizes[dim]:
            raise ValueError(f"{values.size} quadrature weights for {integrand.sizes[dim]} points along {dim!r}")
        weights = weights * xr.DataArray(values, dims=(dim,))
    return (integrand * weights).sum(list(SPATIAL_DIMS))


def _spatial_array(values, data):
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=SPATIAL_DIMS,
        coords={d: data[d] for d in SPATIAL_DIMS},
    )


def volume_integral(data: xr.DataArray, *, form: int = 0, weight=None, domain=None, quadrature=None) -> xr.DataArray:
    """``int w f dV`` over the logical grid, as a function of every other dimension (e.g. ``t``).

    ``form=0`` (default) is a function, integrated with the volume element ``|sqrt g| de``;
    ``form=3`` is a density (a 3-form, e.g. Struphy's L2 fields), integrated as ``int f de``.
    ``weight`` is an optional ``(eta1, eta2, eta3)`` array. The geometry comes from a struphy
    ``domain`` (``out.domain``, exact) or, without one, from the X, Y, Z coordinates.
    ``quadrature`` maps ``eta1``/``eta2``/``eta3`` to explicit weights (e.g. Gauss weights, see
    ``out.analysis.quadrature_grid()``); by default see :func:`quadrature_weights`.
    """
    if form not in (0, 3):
        raise ValueError("volume_integral takes form=0 (a function) or form=3 (a density)")
    validate_array(data)
    integrand = data
    if form == 0:
        sqrt_g, _ = _geometry(data, domain)
        integrand = integrand * _spatial_array(sqrt_g, data)
    if weight is not None:
        integrand = integrand * np.asarray(weight, dtype=float)
    out = _integrate(integrand, quadrature)
    out.attrs = {**_provenance(data), "label": f"integral of {_label(data)}".strip()}
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
    r"""The quadratic energy ``alpha * 1/2 int w  omega^T A omega de``, as a function of time.

    ``form`` says what ``data`` holds, and sets the metric factor ``A`` (as struphy's mass
    matrices do, so that e.g. LinearMHD's ``en_U`` is ``field_energy(u, form=2, weight=n0)``):

    ========================  ===========================================  ==============
    ``form``                  data                                         ``A``
    ========================  ===========================================  ==============
    ``None`` (default)        a function, or Cartesian vector components    ``|sqrt g|``
    ``0``                     0-form                                        ``|sqrt g|``
    ``1``                     1-form components                             ``G^-1 |sqrt g|``
    ``2``                     2-form components                             ``G / |sqrt g|``
    ``3``                     3-form                                        ``1 / |sqrt g|``
    ``"v"``                   contravariant vector components               ``G |sqrt g|``
    ========================  ===========================================  ==============

    Vectors have a ``component`` dimension. Non-finite ``weight`` values (e.g. ``1/p0`` where
    ``p0`` vanishes on the boundary) are left out. The energy of a filtered field, e.g. from
    :func:`~struphy_plots.spectral.filter_time`, measures how much of the energy is in that mode.
    ``quadrature``: as for :func:`volume_integral`. A spline field squared is integrated exactly
    only with enough points per element: evaluate it at Gauss points for accurate energies.
    """
    validate_array(data)
    if form not in (None, 0, 1, 2, 3, "v"):
        raise ValueError(f"form must be None, 0, 1, 2, 3 or 'v'; got {form!r}")
    sqrt_g, metric = _geometry(data, domain)
    w = np.ones_like(sqrt_g) if weight is None else np.asarray(weight, dtype=float) * np.ones_like(sqrt_g)
    w = np.where(np.isfinite(w), w, 0.0)
    vector = "component" in data.dims
    if vector and data.sizes["component"] != 3:
        raise ValueError(f"a vector field needs 3 components; got {data.sizes['component']}")
    if not vector and form in (1, 2, "v"):
        raise ValueError(f"form={form!r} needs vector components (a 'component' dimension)")
    if vector and form in (0, 3):
        raise ValueError(f"form={form} is a scalar; a vector field needs form None, 1, 2 or 'v'")

    if form in (None, 0):
        factor = _spatial_array(w * sqrt_g, data)
        squared = (data**2).sum("component") if vector else data**2
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
        A = xr.DataArray(
            tensor,
            dims=("component", "component_2", *SPATIAL_DIMS),
            coords={d: data[d] for d in SPATIAL_DIMS},
        )
        other = data.rename(component="component_2").drop_vars("component_2", errors="ignore")
        squared = (data.drop_vars("component", errors="ignore") * A * other).sum(("component", "component_2"))
        factor = 1.0
    out = normalization * 0.5 * _integrate(squared * factor, quadrature)
    out.attrs = {**_provenance(data), "label": f"energy of {_label(data)}".strip()}
    return out


def gradient(data: xr.DataArray, *, domain=None) -> xr.DataArray:
    """The Cartesian gradient of a scalar field on the logical grid, with a ``component``
    dimension ``(x, y, z)`` in front of the field's own dimensions (e.g. ``t`` is kept).

    ``grad f = J^-T d f / d eta``, with the Jacobian ``J`` of the mapping from a struphy
    ``domain`` (exact) or, without one, from the attached ``X``, ``Y``, ``Z`` coordinates. The
    logical derivatives are spectral around periodic directions and second order elsewhere
    (see :func:`struphy_plots.arrays.logical_derivative`). A direction with a single point (a
    2-D run) is left out: the result is then the gradient within the plane. For example,
    ``E = -gradient(phi)``, or the E x B velocity ``z x grad(phi)`` of a 2-D drift-wave model.
    """
    from .arrays import logical_derivative, mapping_jacobian, periodicity

    validate_array(data)
    missing = [d for d in SPATIAL_DIMS if d not in data.dims]
    if missing:
        raise ValueError(f"gradient needs the logical dimensions eta1, eta2, eta3; {missing} are missing")
    if "component" in data.dims:
        raise ValueError("gradient takes a scalar field; select a component first")
    field = data.transpose(..., *SPATIAL_DIMS)
    etas = [np.asarray(field[d], dtype=float) for d in SPATIAL_DIMS]
    varying = [i for i, eta in enumerate(etas) if eta.size > 1]
    if not varying:
        raise ValueError("gradient needs at least one logical direction with more than one point")
    if domain is not None:
        points = np.stack([np.asarray(c, dtype=float) for c in domain(*etas, squeeze_out=False)], axis=-1)
        jacobian = np.asarray(domain.jacobian(*etas), dtype=float)
    else:
        if any(name not in field.coords for name in ("X", "Y", "Z")):
            raise ValueError("gradient needs the X, Y, Z coordinates, or a struphy domain")
        points = np.stack(
            [np.asarray(field.coords[name].transpose(*SPATIAL_DIMS), dtype=float) for name in ("X", "Y", "Z")], axis=-1
        )
        if len(varying) == 3:
            jacobian = mapping_jacobian(field)
        else:
            jacobian = np.zeros((3, 3, *points.shape[:3]))
            for i in varying:
                kind = periodicity(points, i)
                for a in range(3):
                    jacobian[a, i] = logical_derivative(points[..., a], etas[i], i, kind)
    values = np.asarray(field, dtype=float)
    offset = values.ndim - 3
    derivatives = [
        logical_derivative(values, etas[i], offset + i, periodicity(points, i)) for i in varying
    ]
    columns = np.moveaxis(jacobian[:, varying], (0, 1), (-2, -1))  # (n1, n2, n3, 3, k)
    transform = np.swapaxes(np.linalg.pinv(columns), -1, -2)  # pinv(J)^T: (n1, n2, n3, 3, k)
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

    ``args`` names the coordinates passed positionally, in order; by default the physical
    ``X``, ``Y``, ``Z`` (if attached, else the logical dimensions ``eta1``... that ``data``
    has), followed by ``t`` if ``data`` has it. E.g. an exact solution ``exact(x, y, z, t)`` on
    the points of a field, or ``exact(eta1, t)`` on a 1-D profile.
    """
    if args is None:
        space = ["X", "Y", "Z"] if all(c in data.coords for c in ("X", "Y", "Z")) else [d for d in SPATIAL_DIMS if d in data.dims]
        args = [*space, *(["t"] if "t" in data.coords else [])]
    missing = [name for name in args if name not in data.coords]
    if missing:
        raise ValueError(f"{missing} are not coordinates of {data.name!r}; it has {tuple(data.coords)}")
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

    ``exact`` is an array (aligned with ``data``) or a function of its coordinates, evaluated
    with :func:`evaluate_on` (``args`` as there), e.g. ``lambda x, y, z, t: ...``. ``norm`` is
    ``"pointwise"`` (the difference itself), ``"max"``, ``"rms"`` (default), ``"l1"`` or
    ``"l2"``, over ``dims`` (default: every dimension but ``t``): unweighted, the mean over the
    grid points (``l1``, and ``rms`` = ``l2`` its root mean square), i.e. integrals over the
    logical unit cube. ``relative`` divides by the same norm of the exact solution. With
    ``weighted`` the norms over the logical grid are integrals over the physical volume
    (``rms`` divided by that volume), (with ``|sqrt g|``, from ``domain`` or the ``X``, ``Y``, ``Z``
    coordinates; see :func:`volume_integral`): the proper L2 error on a mapped domain, where a
    plain mean over grid points is not.
    """
    validate_array(data)
    if callable(exact):
        reference = evaluate_on(data, exact, args)
    elif isinstance(exact, xr.DataArray):
        reference = exact.broadcast_like(data)
    else:  # a plain array (or number) on the grid of data
        exact = np.asarray(exact)
        reference = data.copy(data=np.broadcast_to(exact, data.shape)) if exact.ndim else xr.full_like(data, float(exact))
    difference = data - reference
    if norm == "pointwise":
        out = difference / abs(reference).max() if relative else difference
        out.attrs = {**_provenance(data), "label": f"error of {_label(data)}".strip()}
        return out
    if norm not in ("max", "rms", "l1", "l2"):
        raise ValueError(f"norm must be 'pointwise', 'max', 'rms', 'l1' or 'l2'; got {norm!r}")
    dims = [d for d in data.dims if d != "t"] if dims is None else ([dims] if isinstance(dims, str) else list(dims))

    def measure(values):
        # unweighted: averages over the grid points, i.e. integrals over the logical unit cube;
        # weighted: integrals over the physical volume (rms divides by that volume)
        if norm == "max":
            return abs(values).max(dims)
        power = 1 if norm == "l1" else 2
        if weighted:
            if not set(SPATIAL_DIMS) <= set(values.dims) or set(dims) != set(SPATIAL_DIMS):
                raise ValueError("weighted norms integrate over eta1, eta2, eta3 (the default dims)")
            integral = volume_integral(abs(values) ** power, domain=domain)
            if norm == "rms":
                ones = xr.ones_like(values.isel({d: 0 for d in values.dims if d not in SPATIAL_DIMS}, drop=True))
                integral = integral / volume_integral(ones, domain=domain)
        else:
            integral = (abs(values) ** power).mean(dims)
        return integral if norm == "l1" else np.sqrt(integral)

    out = measure(difference)
    if relative:
        out = out / measure(reference)
    out.attrs = {**_provenance(data), "label": f"{'relative ' if relative else ''}{norm} error of {_label(data)}".strip()}
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

    ``kind="sin"`` gives ``a`` of ``a sin(2 pi number x / period)``: ``2 <f sin(...)>``;
    ``"cos"`` the cosine amplitude, and ``"complex"`` the complex amplitude ``2 <f exp(-i...)>``
    (``abs()`` the amplitude, ``np.angle()`` the phase). The dimension must sample one full period
    uniformly (a duplicate endpoint is dropped). ``bin_correction`` undoes the damping of binned
    particle data, whose bins average the mode over their width ``h``: it divides by
    ``sinc(number h / period)``. A mode number, not a wavenumber: ``k = 2 pi number / L``.
    """
    from .spectral import drop_periodic_endpoint

    if kind not in ("sin", "cos", "complex"):
        raise ValueError(f"kind must be 'sin', 'cos' or 'complex'; got {kind!r}")
    trimmed = drop_periodic_endpoint(data, dim, period=period)
    x = np.asarray(trimmed[dim], dtype=float)
    h = x[1] - x[0] if x.size > 1 else period
    if x.size < 2 or not np.allclose(np.diff(x), h) or not np.isclose(x.size * h, period, rtol=1e-6):
        raise ValueError(f"{dim!r} must sample one full period ({period}) uniformly")
    phase = 2 * np.pi * number * trimmed[dim] / period
    basis = {"sin": np.sin(phase), "cos": np.cos(phase), "complex": np.exp(-1j * phase)}[kind]
    amplitude = 2 * (trimmed * basis).mean(dim)
    if bin_correction:
        amplitude = amplitude / np.sinc(number * h / period)
    amplitude.name = f"{kind}_{number:g}"
    amplitude.attrs = {**_provenance(data), "label": f"{kind} amplitude of mode {number:g} along {dim}"}
    return amplitude


def _cartesian_vector(vector: xr.DataArray, components: str) -> xr.DataArray:
    if "component" not in vector.dims or vector.sizes["component"] != 3:
        raise ValueError("expected a vector field with a 'component' dimension of size 3")
    if components == "contravariant":
        from .pyvista_plots import push_forward

        return push_forward(vector)
    if components != "cartesian":
        raise ValueError(f'components must be "cartesian" or "contravariant"; got {components!r}')
    return vector


def _jacobian_of_components(vector, domain):
    """``d v_a / d x_b`` as an array over (a, b, ...)."""
    rows = [gradient(vector.isel(component=a, drop=True), domain=domain) for a in range(3)]
    return xr.concat(rows, dim="row")  # (row a, component b, ...)


def divergence(vector: xr.DataArray, *, components: str = "cartesian", domain=None) -> xr.DataArray:
    """The divergence ``sum_a d v_a / d x_a`` of a vector field on a mapped domain.

    ``components`` as for the 3-D views (``"contravariant"`` components are pushed forward
    first). Derivatives as in :func:`gradient`: spectral around periodic angles; a flat
    direction (2-D run) is left out. E.g. a ``div B`` check of an MHD run.
    """
    cartesian = _cartesian_vector(vector, components)
    jac = _jacobian_of_components(cartesian, domain)
    out = sum(jac.isel(row=a, component=a, drop=True) for a in range(3))
    out.name = f"div_{vector.name}" if vector.name else "divergence"
    out.attrs = {**_provenance(vector), "label": f"divergence of {_label(vector)}".strip()}
    return out


def curl(vector: xr.DataArray, *, components: str = "cartesian", domain=None) -> xr.DataArray:
    """The curl of a vector field on a mapped domain, in Cartesian components ``(x, y, z)``.

    As :func:`divergence`. E.g. the current ``J = curl B``, or the vorticity of a flow; for a
    2-D field in the ``x``-``y`` plane only the ``z`` component is non-zero.
    """
    cartesian = _cartesian_vector(vector, components)
    jac = _jacobian_of_components(cartesian, domain)

    def d(a, b):
        return jac.isel(row=a, component=b, drop=True)

    out = xr.concat([d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)], dim="component")
    out = out.assign_coords(component=[0, 1, 2]).transpose("component", ...)
    out.name = f"curl_{vector.name}" if vector.name else "curl"
    out.attrs = {**_provenance(vector), "label": f"curl of {_label(vector)}".strip()}
    return out


def flux_function(vector: xr.DataArray) -> xr.DataArray:
    """The flux function (or stream function) ``A`` of a 2-D, divergence-free in-plane field.

    For a magnetic field ``B_x = dA/dy``, ``B_y = -dA/dx`` (so ``B = grad A x z``); its contour
    lines are the field lines, e.g. drawn over the current with a slice's ``contours_of``. For a
    velocity it is the stream function. The field must lie on a Cartesian grid in the ``x``-``y``
    plane (physical ``X`` depending on one logical direction and ``Y`` on another, e.g. a slab
    or periodic box); ``A`` is integrated along the grid lines (trapezoidal rule) and shifted to
    zero mean, as a function of every other dimension.
    """
    cartesian = _cartesian_vector(vector, "cartesian").isel(component=[0, 1])
    field = cartesian.squeeze([d for d in SPATIAL_DIMS if d in cartesian.dims and cartesian.sizes[d] == 1], drop=False)
    grid = [d for d in SPATIAL_DIMS if d in field.dims and field.sizes[d] > 1]
    if len(grid) != 2 or not all(c in field.coords for c in ("X", "Y")):
        raise ValueError("flux_function needs a 2-D field on two logical directions with X, Y coordinates")
    X = np.asarray(field.coords["X"].squeeze(drop=True).transpose(*grid), dtype=float)
    Y = np.asarray(field.coords["Y"].squeeze(drop=True).transpose(*grid), dtype=float)
    if np.allclose(X, X[:, :1]) and np.allclose(Y, Y[:1, :]):
        xdim, ydim = grid
    elif np.allclose(X, X[:1, :]) and np.allclose(Y, Y[:, :1]):
        ydim, xdim = grid
    else:
        raise ValueError("flux_function needs a Cartesian grid: X along one logical direction and Y along the other")
    x = np.asarray(field.coords["X"].isel({ydim: 0}).squeeze(drop=True), dtype=float)
    y = np.asarray(field.coords["Y"].isel({xdim: 0}).squeeze(drop=True), dtype=float)
    bx = field.isel(component=0, drop=True).transpose(..., xdim, ydim)
    by = field.isel(component=1, drop=True).transpose(..., xdim, ydim)

    def cumulative(values, coordinate, axis):
        steps = np.diff(coordinate)
        shape = [1] * values.ndim
        shape[axis] = steps.size
        pairs = 0.5 * (np.take(values, range(1, values.shape[axis]), axis=axis) + np.take(values, range(values.shape[axis] - 1), axis=axis))
        integral = np.cumsum(pairs * steps.reshape(shape), axis=axis)
        return np.concatenate([np.zeros_like(np.take(values, [0], axis=axis)), integral], axis=axis)

    bx_values, by_values = np.asarray(bx, dtype=float), np.asarray(by, dtype=float)
    # A(x, y) = int_0^y B_x(x0, y') dy' - int_0^x B_y(x', y) dx'
    along_y = cumulative(np.take(bx_values, [0], axis=-2), y, -1)
    along_x = cumulative(by_values, x, -2)
    values = along_y - along_x
    values = values - values.mean(axis=(-2, -1), keepdims=True)
    out = bx.copy(data=values)
    out = out.transpose(*[d for d in vector.dims if d != "component" and d in out.dims])
    out.name = "flux_function"
    out.attrs = {**_provenance(vector), "label": f"flux function of {_label(vector)}".strip()}
    return out


def _angles(data: xr.DataArray, R0: float = 0.0, Z0: float = 0.0):
    X, Y, Z = (data.coords[n] for n in ("X", "Y", "Z"))
    R = np.hypot(X, Y)
    phi = np.arctan2(Y, X)
    theta = np.arctan2(Z - Z0, R - R0)
    return R, phi, theta


def cylindrical_components(vector: xr.DataArray) -> xr.DataArray:
    """Cartesian components rotated to cylindrical ones ``(R, phi, Z)`` about the ``Z`` axis,
    at every point of the field (needs its ``X``, ``Y``, ``Z`` coordinates)."""
    cartesian = _cartesian_vector(vector, "cartesian")
    _, phi, _ = _angles(cartesian)
    vx, vy, vz = (cartesian.isel(component=i, drop=True) for i in range(3))
    out = xr.concat(
        [vx * np.cos(phi) + vy * np.sin(phi), -vx * np.sin(phi) + vy * np.cos(phi), vz], dim="component"
    ).assign_coords(component=["R", "phi", "Z"]).transpose(*vector.dims)
    out.name, out.attrs = vector.name, {**vector.attrs, "label": f"{_label(vector)} (R, phi, Z)"}
    return out


def toroidal_components(vector: xr.DataArray, *, R0: float, Z0: float = 0.0) -> xr.DataArray:
    """Cartesian components rotated to the local ``(radial, poloidal, toroidal)`` directions
    about a circular magnetic axis at major radius ``R0`` (height ``Z0``), at every point.

    The radial direction points away from the axis in the poloidal plane, the poloidal one
    along ``theta = atan2(Z - Z0, R - R0)``, the toroidal one along ``phi``: the natural
    components for waves in a tokamak or torus.
    """
    cartesian = _cartesian_vector(vector, "cartesian")
    _, phi, theta = _angles(cartesian, R0, Z0)
    vx, vy, vz = (cartesian.isel(component=i, drop=True) for i in range(3))
    v_R = vx * np.cos(phi) + vy * np.sin(phi)
    v_phi = -vx * np.sin(phi) + vy * np.cos(phi)
    out = xr.concat(
        [v_R * np.cos(theta) + vz * np.sin(theta), -v_R * np.sin(theta) + vz * np.cos(theta), v_phi], dim="component"
    ).assign_coords(component=["radial", "poloidal", "toroidal"]).transpose(*vector.dims)
    out.name, out.attrs = vector.name, {**vector.attrs, "label": f"{_label(vector)} (radial, poloidal, toroidal)"}
    return out


def polar_coordinates(data: xr.DataArray, *, center=(0.0, 0.0)) -> xr.DataArray:
    """``data`` with coordinates ``r`` and ``theta`` (radians, in ``(-pi, pi]``) of its points
    in the ``X``-``Y`` plane about ``center``, e.g. for profiles against the radius."""
    X, Y = data.coords["X"], data.coords["Y"]
    return data.assign_coords(r=np.hypot(X - center[0], Y - center[1]), theta=np.arctan2(Y - center[1], X - center[0]))


def orbit_invariants(orbits: xr.Dataset, *, absB=None) -> xr.Dataset:
    """Kinetic invariants of saved marker orbits, over ``(t, marker)``: whatever the saved
    quantities allow.

    * ``speed`` ``|v|`` from ``v1``, ``v2``, ``v3`` (full orbits);
    * with ``v_par``, ``mu`` and ``absB`` (a function ``|B|(x, y, z)`` of the physical positions,
      e.g. ``lambda x, y, z: out.equil.absB0(*out.domain.inverse_map(x, y, z))``): the
      guiding-centre ``energy`` ``v_par^2 / 2 + mu |B|`` and the ``pitch`` ``v_par / v``.

    Their drift, e.g. ``relative_error()`` of the energy, measures the pusher's accuracy.
    Samples where a marker has left the domain are NaN.
    """
    from .plotting import _alive

    alive = xr.DataArray(_alive(orbits.transpose("t", "marker", ...)), dims=("t", "marker"))
    out = {}
    if all(n in orbits for n in ("v1", "v2", "v3")):
        out["speed"] = np.sqrt(orbits.v1**2 + orbits.v2**2 + orbits.v3**2)
    if absB is not None and all(n in orbits for n in ("v_par", "mu", "x", "y", "z")):
        B = xr.DataArray(np.asarray(absB(np.asarray(orbits.x), np.asarray(orbits.y), np.asarray(orbits.z)), dtype=float),
                         dims=orbits.x.dims, coords=orbits.x.coords)
        energy = 0.5 * orbits.v_par**2 + orbits.mu * B
        out["energy"] = energy
        out["pitch"] = orbits.v_par / np.sqrt(2 * energy)
    if not out:
        raise ValueError("no invariant can be computed from these orbits (need v1..v3, or v_par, mu and absB)")
    return xr.Dataset({name: values.where(alive) for name, values in out.items()}, attrs=_provenance(orbits))


def bounce_period(orbits: xr.Dataset, *, v_par: str = "v_par") -> xr.DataArray:
    """The bounce period of each trapped marker: twice the mean time between reversals of its
    parallel velocity (NaN for markers with fewer than two reversals, e.g. passing ones)."""
    velocity = orbits[v_par].transpose("t", "marker")
    t = np.asarray(orbits.t, dtype=float)
    periods = []
    for m in range(velocity.sizes["marker"]):
        v = np.asarray(velocity.isel(marker=m))
        sign = np.sign(v)
        flips = np.flatnonzero((sign[1:] * sign[:-1]) < 0)
        # linear interpolation of the crossing times
        crossings = t[flips] - v[flips] * (t[flips + 1] - t[flips]) / (v[flips + 1] - v[flips])
        periods.append(2 * np.mean(np.diff(crossings)) if crossings.size >= 2 else np.nan)
    out = xr.DataArray(np.asarray(periods), dims="marker", coords={"marker": orbits.marker}, name="bounce_period")
    out.attrs = {**_provenance(orbits), "label": "bounce period"}
    return out
