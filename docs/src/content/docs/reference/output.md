---
title: OutputPlots
description: API reference for whole-run plots.
---

`struphy_plots.output_accessors.OutputPlots`, accessed as
`output.struphy.plot` on a Struphy `Output` object (not a `DataArray`).

### `scalars(names=None, *, relative_to=None, logy=False)`
Overview of recorded scalar time series. `names` restricts to a subset;
`relative_to` normalizes each series against a reference (e.g. `"initial"`).

### `equilibrium(ax=None)`
Radial equilibrium profiles read from the run's `geometry.vts`.

### `equilibrium_3d(*, scalars="p0", cmap="viridis")`
Interactive 3-D equilibrium view via PyVista. Requires
`pip install "struphy-plots[pyvista]"`.
