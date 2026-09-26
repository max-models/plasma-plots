---
title: Whole-run plots
description: Scalar overviews and equilibrium plots for an entire Struphy run.
---

`OutputPlots`, from `struphy_plots.output_accessors`, works on a Struphy
`Output` object rather than a single array, and gives you overview plots for
a whole run. Importing `struphy_plots` also wires it up as `out.plot`, so
`out.plot()` is a quick default (the scalar overview).

```python
from struphy.post_processing.output import Output
out = Output("path/to/run")

out.struphy.plot.scalars(relative_to="initial", logy=True)
out.struphy.plot.equilibrium()
out.struphy.plot.equilibrium_3d(scalars="p0", cmap="viridis")
```

- **`scalars(names=None, *, relative_to=None, logy=False)`** — plots every
  recorded scalar time series (e.g. field/kinetic energies) in one figure.
  Pass `names` to restrict to a subset, or `relative_to` to normalize each
  series against its initial value.

  ![Overview of every scalar time series in one run](../../../assets/figures/scalars.png)

- **`equilibrium(ax=None)`** — radial equilibrium profiles read from the
  run's `geometry.vts`.

  ![Radial equilibrium profiles: pressure, density, temperature](../../../assets/figures/equilibrium.png)

- **`equilibrium_3d(scalars="p0", cmap="viridis")`** — interactive 3-D
  equilibrium view via PyVista. Requires
  `pip install "struphy-plots[pyvista]"`.

  ![A 3-D equilibrium view shaded by a scalar field](../../../assets/figures/equilibrium_3d.png)

See the [Whole-run reference](/struphy-plots/reference/output/) for full
signatures.
