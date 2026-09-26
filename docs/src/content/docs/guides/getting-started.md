---
title: Getting started
description: Install struphy-plots and make your first plot.
---

## Installation

```bash
pip install struphy-plots
```

For interactive 3-D volume rendering, install the optional PyVista extra:

```bash
pip install "struphy-plots[pyvista]"
```

## Registering the accessor

Importing `struphy_plots` registers the `.struphy` accessor on every
`xarray.DataArray`:

```python
import struphy_plots  # registers DataArray.struphy

field = out.evaluate("em_fields/phi")
field.struphy.plot.slice(x="e1", y="e2", t="last")
```

You don't need to keep a reference to `struphy_plots` around — the import
alone is enough to enable `.struphy` on any array produced by Struphy's
`Output` object.

## Two entry points

- **`array.struphy.plot`** — plotting methods for a single labeled array (see
  [Plotting](/struphy-plots/guides/plotting/)).
- **`array.struphy.analysis`** — numerical diagnostics on a single labeled
  array (see [Analysis](/struphy-plots/guides/analysis/)).
- **`struphy_plots.output_accessors.OutputPlots`** — overview plots for a
  whole simulation run (see [Whole-run plots](/struphy-plots/guides/output-plots/)).

Lower-level, function-based versions of everything above are also available
directly from `struphy_plots.plotting` and `struphy_plots.analysis`, if you'd
rather call a function on a plain `xarray.DataArray` than go through the
accessor.
