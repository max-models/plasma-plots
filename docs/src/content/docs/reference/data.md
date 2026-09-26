---
title: struphy.data
description: API reference for array.struphy.data and dataset.struphy.data.
---

The data behind each plot, without rendering it — every method here mirrors
one on [`array.struphy.plot`](/struphy-plots/reference/plot/) (or
`dataset.struphy.plot`) and returns the same already-selected `xarray`
object it would have drawn, useful to hand to a different plotting library
(Plotly, bokeh, ...) or to inspect directly. See
[Getting the data instead of a plot](/struphy-plots/guides/getting-started/#getting-the-data-instead-of-a-plot)
for examples. Backed by `struphy_plots.accessors.ArrayData` and
`DatasetData`.

## `array.struphy.data`

### `lineout(*, x=None, **selection)`
The 1-D profile `plot.lineout(...)` would draw.

### `vector(*, x, y, components=(0, 1), stride=1, coordinates="logical", **selection)`
The selected, strided vector field `plot.vector(...)` would draw.

### `volume_slices(*, indices=None, **selection)`
The three orthogonal planes `plot.volume_slices(...)` would draw, as a dict
keyed by the fixed dimension (`"eta1"`, `"eta2"`, `"eta3"`).

### `compare(other, *, mode="difference")`
The aligned difference or ratio `plot.compare(...)` would draw.

### `view(*, x=None, y=None, sweep="t", coords="logical", plane="XY", **selection)`
Every remaining dimension, sweep included — the shared data behind
`plot.panels(...)`, `.viewer(...)`, `.animation(...)` and `.frames(...)`,
which each render one frame of exactly this at a time.

### `slice(*, x=None, y=None, sweep="t", coords="logical", plane="XY", **selection)`
The single 2-D slice `plot.slice(...)` would draw.

### `overlay_orbits(orbits, *, x, y, max_markers=200, **selection)`
A `(field_slice, orbit_subset)` tuple: the field slice and marker-position
subset `plot.overlay_orbits(...)` would draw.

### `dispersion(*, dim=None, detrend=True)`
The `(omega, k)` power spectrum `plot.dispersion(...)` would draw. Same as
`array.struphy.analysis.dispersion(...)`; included here too for parity with
every other plot.

### `trajectories(*, max_markers=200)`
The marker-position subset `plot.trajectories(...)` would draw, as an
`xarray.Dataset`.

### `timeseries(*others)`
This time series and any `others`, validated — the same list
`plot.timeseries(...)` would draw.

### `grid(*, name=None, **selection)`
This field as a `pyvista.StructuredGrid` on its physical `X`, `Y`, `Z`
points: the data behind every 3-D view, ready for any PyVista filter. Vector
fields also get their magnitude as `"|name|"`.

### `to_vtk(path, *, name=None, **selection)`
Writes the field to VTK structured grids for ParaView: one `.vts` per time
plus a `.pvd` collection in the directory `path`, or a single `.vts` without
`t`. Backed by `struphy_plots.pyvista_plots.save_vtk`.

### `slices_3d(*, cuts=None, **selection)`
The list of logical cuts `plot.slices_3d(...)` would draw, each an
`xarray.DataArray` that keeps its size-one cut dimension.

## `dataset.struphy.data`

### `trajectories(*, max_markers=200)`
Same as the `DataArray` accessor's `trajectories()` above.

### `scatter(*, x, y, color=None, **selection)`
The selected `xarray.Dataset` `plot.scatter(...)` would draw —
`.to_dataframe()` hands it straight to e.g. Plotly Express.
