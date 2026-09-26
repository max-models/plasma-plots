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

### `dispersion(*, component=0, slice_at=(None, 0, 0), physical=False, **kwargs)`
Space-time power spectrum and dispersion-relation fit. Requires normalized
time and the main `struphy` package (`struphy.post_processing.spectral`).
