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
