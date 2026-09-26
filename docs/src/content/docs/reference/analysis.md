---
title: struphy.analysis
description: API reference for array.struphy.analysis.
---

Accessed as `array.struphy.analysis` on any labeled `xarray.DataArray`.
Backed by `struphy_plots.accessors.ArrayAnalysis`; underlying implementations
live in `struphy_plots.analysis`.

### `growth_rate(*, window=(None, None), amplitude=False)`
Exponential fit over `window`; returns the fitted rate (and amplitude if
`amplitude=True`).

### `damping_rate(*, window=(None, None), amplitude=False)`
Exponential decay fit of the signal's envelope.

### `envelope()`
Local maxima of the time series, as a new `xarray.DataArray`.

### `norm(*, dims=None, squared=False)`
L2 norm over `dims` (defaults to all spatial dims).

### `drift(*, ref=None)`
Deviation from a reference array, or from the first sample along `t` if
`ref` is not given.

### `relative_error(*, ref=None, skip_first=True)`
Relative deviation from a reference array.

### `spatial_average(*, dims=None)`
Mean over the logical space dimensions.

### `velocity_moments(*, dims=None)`
Density, mean velocity, and variance over velocity dimensions. Returns an
`xarray.Dataset`.

### `dispersion(*, dim=None, detrend=True)`
The `(omega, k)` space-time power spectrum of this `(t, dim)` field — a
detrended 2-D FFT, independent of Struphy. `dim` defaults to the sole
dimension other than `t`. See `struphy_plots.analysis.power_spectrum` for
the definition, and `ArrayPlots.dispersion` to plot it directly.

### `gradient(*, domain=None)`
The Cartesian gradient `J⁻ᵀ ∂f/∂η` of a scalar field, with a `component`
dimension (x, y, z). The Jacobian comes from `X`, `Y`, `Z` or from a Struphy
`domain`. Derivatives are spectral around periodic angles. For a 2-D run it is
the in-plane gradient. Backed by `struphy_plots.analysis.gradient`.

### Spectral diagnostics
`fft(dim=...)`, `time_fft()`, `filter_time()`, `band_filter(lo, hi)`,
`spectral_peaks()`, `spectrogram(length=...)`, `mode_spectrum()`,
`mode_amplitudes()`, `mode_structure(omega)`, `cross_spectrum(other)`,
`matrix_pencil(n_modes=...)` and `drop_periodic_endpoint(dim)`. See the
[spectral reference](/struphy-plots/reference/spectral/).

## `dataset.struphy.analysis` (Dataset accessor)

### `classify_orbits(*, v_par="v_par")`
For a guiding-center orbits product: each marker's class as a `(marker,)`
array of integer codes, passing (0), trapped (1) or lost (-1). These are
the same criteria as Struphy's `post_process_orbit_classification`. A
marker is trapped if `v_par` ever has the opposite sign to its initial
value. It is lost if every saved quantity is zero at some time, which is
how Struphy stores a marker that has left the domain. Lost takes precedence
over trapped. The class names are in `attrs["flag_meanings"]`.

## Standalone functions

### `struphy_plots.analysis.volume_integral(data, *, form=0, weight=None, domain=None, quadrature=None)`
`∫ w f dV` over the logical grid, as a function of the other dimensions.
`form=0` is a function (integrated with |√g|), `form=3` a density. The
geometry comes from a Struphy `domain` or from the `X`, `Y`, `Z` coordinates.

### `struphy_plots.analysis.field_energy(data, *, form=None, weight=None, domain=None, normalization=1.0, quadrature=None)`
`α · ½ ∫ w ωᵀ A ω dη`. `A` follows `form`: `None`/`0` for a function or
Cartesian components (|√g|), `1`, `2`, `3` for p-forms (`G⁻¹|√g|`, `G/|√g|`,
`1/|√g|`), `"v"` for contravariant components (`G|√g|`), as in Struphy's mass
matrices.

### `struphy_plots.analysis.quadrature_weights(coordinate)`
Midpoint weights for Struphy's cell centers, trapezoidal otherwise, and 1 for a
single point.

### `struphy_plots.analysis.convergence_order(sizes, errors)`
Fits `error = constant * size**order` in log-log space, for a convergence
study (`sizes` a resolution or step size, `errors` the corresponding error
norms). Returns a `ConvergenceFit` (`.order`, `.constant`, `.sizes`,
`.fitted`), or `None` with fewer than two valid (finite, positive) samples.
Used by `struphy_plots.plotting.plot_convergence`.

### `struphy_plots.analysis.power_spectrum(data, *, dim=None, detrend=True)`
The 2-D power spectrum of a `(t, dim)` signal, as a function of angular
frequency `omega` and wavenumber `k` — a plain FFT (detrended by default),
the basis of a dispersion-relation plot. Returns an `xarray.DataArray` with
dims `(omega, k)`.
