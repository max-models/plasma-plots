---
title: Field plots
description: Slices, panels, animations, vector fields, and 3-D volumes.
---

Everything here is a method of `array.struphy.plot`, where `array` is a labeled
`xarray.DataArray` produced by Struphy's `Output.evaluate(...)`. Full
signatures are in the [Plotting reference](/struphy-plots/reference/plot/).

## 1-D: lineouts

```python
field.struphy.plot.lineout(x="e1", t="last", e2=0.5)
```

Selects every dimension except `x` (via keyword selection) and plots the
remaining 1-D profile.

![Lineout of a scalar field along one axis](../../../assets/figures/lineout.png)

## 2-D: slices, panels, viewers, animations

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

:::tip[Plot it with Plotly instead]
`.data.slice(...)` returns the exact same selected, transposed array
`.plot.slice(...)` draws — feed it to `px.imshow` for an interactive
heatmap (see [Getting started](/struphy-plots/guides/getting-started/#getting-the-data-instead-of-a-plot)
for why the extra `.transpose(...)` is needed):

```python
import plotly.express as px

selected = field.struphy.data.slice(x="e1", y="e2", t="last")
px.imshow(selected.transpose("e2", "e1"), x=selected.e1, y=selected.e2, origin="lower")
```
:::

`plot.animation(...)` returns a `matplotlib.animation.FuncAnimation`; save it
with `anim.save("orbit.gif", writer="pillow")`:

![An animation swept over time, saved as a GIF](/struphy-plots/figures/animation.gif)

## Vector fields

```python
b_field.struphy.plot.vector(x="e1", y="e2", components=(0, 1), stride=4)
```

![A quiver plot of two vector components](../../../assets/figures/vector.png)

:::tip[Plot it with Plotly instead]
Plotly has no built-in quiver trace; `plotly.figure_factory.create_quiver`
takes flat position and component arrays instead of a grid, so flatten
`.data.vector(...)`'s coordinates and components first:

```python
import numpy as np
import plotly.figure_factory as ff

selected = b_field.struphy.data.vector(x="e1", y="e2", components=(0, 1), stride=4)
xg, yg = np.meshgrid(selected.e1, selected.e2, indexing="ij")
ff.create_quiver(
    xg.ravel(), yg.ravel(),
    selected.isel(component=0).values.ravel(), selected.isel(component=1).values.ravel(),
    scale=0.05,
)
```
:::

## 3-D scalar volumes

Three orthogonal midpoint slices need nothing extra:

```python
density.struphy.plot.volume_slices()
```

![Three orthogonal slices through a 3-D scalar field](../../../assets/figures/volume_slices.png)

:::tip[Plot it with Plotly instead]
`.data.volume_slices()` returns the same three planes as a
`dict[str, xarray.DataArray]`, keyed by the fixed dimension — drop each
straight into its own `go.Heatmap`:

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

planes = density.struphy.data.volume_slices()
fig = make_subplots(rows=1, cols=3, subplot_titles=list(planes))
for i, (name, plane) in enumerate(planes.items(), start=1):
    x, y = plane.dims
    fig.add_trace(go.Heatmap(z=plane.values.T, x=plane[x].values, y=plane[y].values), row=1, col=i)
```
:::

A full interactive volume render needs the optional PyVista extra
(`pip install "struphy-plots[pyvista]"`):

```python
plotter = density.struphy.plot.volume(cmap="viridis", opacity="linear")
plotter.show()
```

![A PyVista volume render of a 3-D scalar field](../../../assets/figures/volume.png)

## Comparing two arrays

```python
field.struphy.plot.compare(reference_field, mode="difference")
```

See [Time series & comparisons](/struphy-plots/guides/timeseries/) for
`compare()` figures — it's a 1-D lineout under the hood, so it fits either
page; we keep the write-up there next to `timeseries()`.
