---
title: Plotting
description: Time series, lineouts, slices, panels, vectors, animations and volumes.
---

All plotting methods live on `array.struphy.plot`, where `array` is a labeled
`xarray.DataArray` produced by Struphy's `Output.evaluate(...)`. Full
signatures are in the [Plotting reference](/struphy-plots/reference/plot/).

## Time series

```python
energy.struphy.plot.timeseries(logy=True, fit="growth_rate")
```

Overlays an optional exponential growth/damping fit on top of one or more
scalar time series.

## Lineouts

```python
field.struphy.plot.lineout(x="e1", t="last", e2=0.5, e3=0.0)
```

Selects every dimension except `x` (via keyword selection) and plots the
remaining 1-D profile.

![Lineout of a scalar field along one axis](../../../assets/figures/lineout.png)

## 2-D slices, panels, and animations

`plot.view(...)` configures a reusable 2-D slice — pick the two axes to plot
(`x`, `y`) and which dimension to sweep over (`sweep`, default `"t"`):

```python
view = field.struphy.plot.view(x="e1", y="e2", sweep="t")

view.slice(t="last")              # one snapshot
view.panels(nrows=3, ncols=4)     # a grid of snapshots
view.viewer()                     # interactive slider (Jupyter/IPython)
view.animation(interval=100)      # matplotlib.animation.FuncAnimation
view.save_frames("frames/", step=2, prefix="phi")
```

`plot.slice(...)`, `plot.panels(...)`, `plot.viewer(...)`, `plot.animation(...)`
and `plot.frames(...)` are shortcuts that build a view and immediately render
it, taking the same keyword arguments as `view(...)`.

![A single 2-D slice snapshot](../../../assets/figures/slice.png)

![A grid of snapshots swept over time](../../../assets/figures/panels.png)

## Vector fields

```python
b_field.struphy.plot.vector(x="e1", y="e2", components=(0, 1), stride=4)
```

![A quiver plot of two vector components](../../../assets/figures/vector.png)

## Comparing two arrays

```python
field.struphy.plot.compare(reference_field, mode="difference")
```

![Ratio of a diagnostic between two runs](../../../assets/figures/compare.png)

## 3-D volumes (optional PyVista)

```python
field.struphy.plot.volume(cmap="viridis", opacity="linear")
```

Returns a PyVista plotter — call `.show()` on it to render. Requires
`pip install "struphy-plots[pyvista]"`.

## Marker trajectories

```python
markers.struphy.plot.trajectories(max_markers=200)
```

Plots 3-D orbits for kinetic marker output; works directly on the
`xarray.Dataset` an orbits product now is (one `(t, marker)` variable per
saved quantity), or on the older `(t, marker, quantity)` `DataArray` form.

![3-D marker trajectories](../../../assets/figures/trajectories.png)
