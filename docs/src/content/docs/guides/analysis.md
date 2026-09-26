---
title: Analysis
description: Growth rates, envelopes, norms, moments and dispersion relations.
---

Numerical diagnostics live on `array.struphy.analysis`, alongside plotting on
the same accessor. Full signatures are in the
[Analysis reference](/struphy-plots/reference/analysis/).

## Growth and damping rates

```python
rate, amplitude = energy.struphy.analysis.growth_rate(window=(0.0, 5.0))
rate, amplitude = energy.struphy.analysis.damping_rate()
```

Both fit an exponential to the (enveloped, for damping) time series and
return the fitted rate. `array.struphy.plot.timeseries(fit=...)` draws the
same fit on top of the series:

![A time series with an exponential growth-rate fit overlaid](../../../assets/figures/timeseries.png)

## Norms, drift, and error

```python
field.struphy.analysis.norm(squared=True)
field.struphy.analysis.drift(ref=field.isel(t=0))
field.struphy.analysis.relative_error(ref=exact_solution)
```

## Spatial averages and velocity moments

```python
field.struphy.analysis.spatial_average()
distribution.struphy.analysis.velocity_moments()
```

`velocity_moments` returns an `xarray.Dataset` with density, mean velocity,
and variance computed over the velocity dimensions.

## Dispersion relations

```python
field.struphy.analysis.dispersion(component=0, slice_at=(None, 0, 0))
```

Computes a space-time power spectrum and fits a dispersion relation. Requires
normalized time and pulls in `struphy.post_processing.spectral`, so it needs
the main `struphy` package installed alongside `struphy-plots`.
