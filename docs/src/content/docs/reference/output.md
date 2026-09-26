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
Radial profiles of the run's fluid equilibrium (`out.equil`, `out.domain`):
`p0`, and `n0`/`T0 = p0/n0` if the equilibrium has a density profile.

### `equilibrium_3d(*, scalars="p0", cmap="viridis")`
Interactive 3-D equilibrium view via PyVista. Requires
`pip install "struphy-plots[pyvista]"`.

### `domain_3d(*, n1=8, n2=32, n3=32, surface=True)`
A PyVista wireframe of the run's mapping (`out.domain`): grid lines on the
real boundary and on the `e3 = 0` cross-section. See
`struphy_plots.pyvista_plots.pyvista_domain` for more options.

### `profile`
A `ProfilePlots`, wrapping `out.profile.results` (a
`scope_profiler.ProfilingResults`, from a run started with
`sim.run(profiling_activated=True)`). Requires
`pip install "struphy-plots[profiling]"`. See the
[Profiling guide](/struphy-plots/guides/profiling/) for figures.

## `ProfilePlots`

`struphy_plots.output_accessors.ProfilePlots`, accessed as
`output.struphy.plot.profile`. Each method is a thin pass-through to the
matching [scope-profiler](https://pypi.org/project/scope-profiler/)
plotting function on `output.profile.results`, forwarding every other
keyword argument straight through (`ranks`, `include`/`exclude`,
`backend`, `filepath`, ...) and defaulting `return_fig=True` and
`verbose=False`. Returns exactly what scope-profiler itself returns: a
`(fig, axes)` pair for the default matplotlib backend, or a Plotly figure
with `backend="plotly"`.

### `gantt(**kwargs)`
A timeline of every recorded region, one row per rank.

### `flame(**kwargs)`
A flame chart reconstructing the call stack from region timings.

### `callgraph(**kwargs)`
The explicit call graph (which region calls which), without timings. Pass
`compact=True` to collapse every invocation of a region into one node.

## `OutputAnalysis`

`struphy_plots.output_accessors.OutputAnalysis`, accessed as `out.analysis`.
`fft(product, *, dim, ...)`, `time_fft(product, ...)`,
`filter_time(product, ...)` and `mode_spectrum(product, ...)` take a product
name (evaluated with `out.evaluate`) or an array. See the
[spectral reference](/struphy-plots/reference/spectral/).
