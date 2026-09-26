---
title: struphy.plot
description: API reference for array.struphy.plot.
---

Accessed as `array.struphy.plot` on any labeled `xarray.DataArray`. Backed by
`struphy_plots.accessors.ArrayPlots`; underlying implementations live in
`struphy_plots.plotting` and can be called directly on a plain
`xarray.DataArray` if you prefer not to use the accessor.

### `timeseries(*others, logy=True, fit=None, fit_amplitude=False, title=None, ax=None)`
Plot this time series, optionally alongside `others`, with an optional
exponential growth/damping fit overlay.

### `lineout(*, x=None, ax=None, title=None, **selection)`
Select every dimension via `**selection` except `x`, and plot the remaining
1-D profile.

### `vector(*, x, y, components=(0, 1), stride=1, coordinates="logical", ax=None, **selection)`
Quiver plot of two vector components over the `x`/`y` plane.

### `volume_slices(*, indices=None, cmap=None, **selection)`
Three orthogonal midpoint slices through a scalar volume.

### `volume(*, name=None, cmap="viridis", opacity="linear", **selection)`
PyVista volume plot of a scalar field. Returns a plotter; call `.show()`.

### `compare(other, *, mode="difference", ax=None)`
1-D aligned comparison of this array against `other` (`mode`: `"difference"`
or `"ratio"`).

### `overlay_orbits(orbits, *, x, y, max_markers=200, ax=None, cmap=None, **selection)`
This field's 2-D slice with marker paths from an orbits-like `orbits`
Dataset overlaid — a Poincare-style diagnostic for checking particle
confinement or orbit topology against a background field. `orbits` must have
position variables named `x` and `y` too.

### `view(*, x=None, y=None, sweep="t", coords="logical", plane="XY", vmin=None, vmax=None, shared_clim=True, cmap=None, equal_aspect=None, title=None, **selection)`
Configure a reusable 2-D slice view without rendering it. Returns a
`SliceView` with `.slice()`, `.panels()`, `.viewer()`, `.animation()`, and
`.save_frames()`.

### `slice(..., ax=None, **selection)`
Render one 2-D slice snapshot. Same keyword arguments as `view(...)`.

### `panels(..., nrows=3, ncols=4, **selection)`
Render a grid of evenly spaced snapshots swept over `sweep`.

### `viewer(..., **selection)`
Interactive slider-based viewer (Jupyter/IPython).

### `animation(..., interval=100, step=1, **selection)`
`matplotlib.animation.FuncAnimation` sweeping over `sweep`.

### `frames(directory, ..., step=1, prefix="frame", dpi=110, **selection)`
Export one PNG per swept step into `directory`.

### `trajectories(*, max_markers=200, show_paths=None, ax=None)`
3-D marker-trajectory plot for kinetic orbit output.

## `dataset.struphy.plot` (Dataset accessor)

Accessed as `dataset.struphy.plot` on any `xarray.Dataset` with per-marker
variables, e.g. an orbits product. Backed by
`struphy_plots.accessors.DatasetPlots`.

### `trajectories(*, max_markers=200, show_paths=None, ax=None)`
Same as the `DataArray` accessor's `trajectories()` above.

### `scatter(*, x, y, color=None, ax=None, cmap=None, s=8, **selection)`
Scatter two position variables (`x`, `y`), optionally colored by a third
(e.g. a density, weight, or Lagrangian tracer). Remaining dimensions such as
`t` are selected by keyword, exactly like `lineout()`.

## Standalone functions

`struphy_plots.plotting` also exposes functions that don't hang off a single
array or dataset:

### `plot_convergence(sizes, errors, *, ax=None, order=None, label=None, xlabel="resolution", title="Convergence")`
Log-log plot of an error norm against resolution or step size. With
`order=None` (default), fits and draws the observed order via
`struphy_plots.analysis.convergence_order`; pass an explicit `order` to draw
a reference slope instead.
