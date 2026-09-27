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

## Vector fields

```python
b_field.struphy.plot.vector(x="e1", y="e2", components=(0, 1), stride=4)
```

## Comparing two arrays

```python
field.struphy.plot.compare(reference_field, mode="difference")
```

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

Plots 3-D orbits for kinetic marker output.

## Running under MPI

A post-processing script can run on several ranks, e.g. next to a simulation
with `mpirun -n 4 python script.py`. Plots are then drawn on rank 0 only; the
other ranks get a `SkippedPlot` placeholder whose methods do nothing, so the
same script works in serial and in parallel without `if rank == 0:` guards:

```python
out.plot.scalars().save("scalars.png")                # written once, by rank 0
field.struphy.plot.frames("frames/", x="eta1", y="eta2")  # [] on ranks > 0
```

Analysis (`array.struphy.analysis.*`) still runs on every rank. The rank is read
from the MPI launcher's environment, or from `mpi4py` if MPI is already
initialized — importing `struphy_plots` never initializes MPI itself. Set
`STRUPHY_MPI=0` to make every process plot. Nothing waits for rank 0: call
`MPI.COMM_WORLD.Barrier()` before other ranks read a file rank 0 wrote.
`struphy_plots.is_plotting_rank()` tells whether the current process draws.
