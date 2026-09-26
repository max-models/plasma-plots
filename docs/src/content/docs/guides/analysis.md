---
title: Diagnostics
description: Growth/damping rates, envelopes, norms, drift, and error.
---

These live on `array.struphy.analysis`, next to plotting on the same
accessor, and return plain `xarray` objects or fit results rather than
figures — plot them however you like. Full signatures are in the
[Analysis reference](/struphy-plots/reference/analysis/).

## Growth and damping rates

```python
rate, amplitude = energy.struphy.analysis.growth_rate(window=(0.0, 5.0))
rate, amplitude = energy.struphy.analysis.damping_rate()
```

`growth_rate` fits an exponential directly; `damping_rate` fits the same
exponential to the signal's **envelope** first (`analysis.envelope()`), which
is what you want for an oscillating-and-decaying signal such as the field
energy in Landau damping — fitting the raw oscillation would give nonsense.

![Damping rate fitted to the envelope of an oscillating, decaying signal](../../../assets/figures/damping.png)

## Norm

```python
field.struphy.analysis.norm(squared=False)
```

L2 norm over every dimension except `t` (or over `dims` explicitly),
producing a `(t,)` series you can pass straight to `.struphy.plot.lineout(x="t")`:

![Field norm decaying in time](../../../assets/figures/norm.png)

## Drift and relative error

```python
field.struphy.analysis.drift(ref=field.isel(t=0))
energy.struphy.analysis.relative_error(ref=exact_solution)
```

`drift` is the signed deviation from a reference (or the first sample);
`relative_error` is the same, divided by the reference. Both return a `(t,)`
series — handy for checking whether a conserved quantity actually stays
conserved:

![Drift of a scalar from its initial value](../../../assets/figures/drift.png)

![Relative energy-conservation error growing over a run](../../../assets/figures/relative_error.png)

## Distribution moments and spatial averaging

`spatial_average()` and `velocity_moments()` work on binned distribution
products (`f(t, e1, v1)` and similar) — see
[Particles & distributions](/struphy-plots/guides/particles/) for those, with
figures.

## Dispersion relations

```python
field.struphy.analysis.dispersion(component=0, slice_at=(None, 0, 0))
```

Computes a space-time power spectrum and fits a dispersion relation, via
`struphy.post_processing.spectral`. This module isn't present on every
struphy version — if you get a `ModuleNotFoundError` here, your installed
struphy doesn't currently ship it.
