---
title: Particles & distributions
description: Phase-space slices, velocity moments, spatial averages, marker trajectories, scatter, and orbit overlays.
---

## Phase-space slices

A binned distribution such as `kinetic_ions/e1_v1_density/f` is just another
labeled array, so every [field plot](/struphy-plots/guides/field-plots/)
works on it directly:

```python
distribution.struphy.plot.slice(x="e1", y="v1", t="last")
```

![A bump-on-tail distribution in (e1, v1) phase space](../../../assets/figures/phase_space.png)

## Velocity moments

```python
moments = distribution.struphy.analysis.velocity_moments()
```

Returns an `xarray.Dataset` with `density` (the zeroth moment), `mean_<dim>`
and `variance_<dim>` for each integrated velocity dimension — functions of
the remaining dimensions (e.g. `(t, e1)` for an `e1_v1` product):

![Density, mean velocity, and variance of a distribution along e1](../../../assets/figures/velocity_moments.png)

## Spatial averaging

```python
distribution.struphy.analysis.spatial_average()
```

Averages over the logical space dimensions, e.g. turning `f(t, e1, v1)` into
`f(t, v1)` — plot the result like any other 2-D field to see the velocity
distribution evolve over a run:

![A velocity distribution's growing beam, averaged over space](../../../assets/figures/spatial_average.png)

## Marker trajectories

```python
markers.struphy.plot.trajectories(max_markers=200)
```

Plots 3-D orbits for kinetic marker output. Works directly on the
`xarray.Dataset` an orbits product now is (one `(t, marker)` variable per
saved quantity: `x`, `y`, `z`, `v1`, `v2`, `v3`, `weight`, ...), or on the
older `(t, marker, quantity)` `DataArray` form.

![3-D marker trajectories](../../../assets/figures/trajectories.png)

## Marker scatter and particle clouds

```python
markers.struphy.plot.scatter(x="x", y="y", color="density", t="last")
```

Scatters two position-like variables from any per-marker Dataset (not just
an orbits product — any Dataset with a `marker` dimension), optionally
colored by a third variable such as a density, weight, or a Lagrangian
tracer a particle carries. Useful for checking a marker loading scheme, or
visualizing an SPH particle cloud:

![An expanding particle cloud, colored by a density-like tracer](../../../assets/figures/marker_scatter.png)

## Orbits over a background field

```python
field.struphy.plot.overlay_orbits(orbits, x="e1", y="e2", t="last")
```

Draws this field's 2-D slice with marker paths from an orbits-like Dataset
overlaid — a Poincare-style diagnostic for checking particle confinement or
orbit topology against a background field (e.g. `|B|` or a flux function in
a poloidal cross-section). `orbits` needs position variables named `x` and
`y` too, matching the field's chosen axes:

![Confined orbits at different radii overlaid on a potential well](../../../assets/figures/orbit_overlay.png)
