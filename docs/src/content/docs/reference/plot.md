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

### `dispersion(*, dim=None, detrend=True, branches=None, log=True, dynamic_range=6.0, kmax=None, omega_max=None, vmin=None, vmax=None, cmap=None, ax=None, title=None)`
The space-time power spectrum of this `(t, dim)` field, as a
dispersion-relation plot (a plain 2-D FFT, independent of Struphy). `dim`
defaults to the sole dimension other than `t`. `branches` overlays named
theoretical curves to compare against — a mapping of label to a callable
`omega(k)`, or an explicit `(k, omega)` pair. Color limits default to the
top `dynamic_range` decades below the peak (with `log=True`), since a
dispersion relation's power spans many orders of magnitude between the
ridge and the rest of the plane.

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

### PyVista 3-D views

These need `pip install "struphy-plots[pyvista]"` and return a
`pyvista.Plotter` (call `.show()` or `.screenshot(path)`). They draw the field
on its physical `X`, `Y`, `Z` points, after selecting every dimension but
`e1`, `e2`, `e3` (and `component`) by keyword. A 2-D field is one where a
spatial dimension has a single point or was selected away. See the
[3-D views guide](/struphy-plots/guides/3d-views/) for figures;
implementations are in `struphy_plots.pyvista_plots`.

#### `isosurface(*, values=5, cmap="viridis", opacity=1.0, clim=None, show_domain=True, title=None, plotter=None, **selection)`
Contour surfaces of a scalar field: `values` levels, or a list of levels. For
a 2-D field, contour lines over the colored plane.

#### `slices_3d(*, cuts=None, cmap="viridis", clim=None, show_domain=True, title=None, plotter=None, **selection)`
Surfaces of constant logical coordinate, drawn in physical space, e.g.
`cuts={"e3": [0, 0.25]}` (poloidal cross-sections) or `cuts={"e1": 0.8}` (a
flux surface). A float is the nearest coordinate, an integer an index,
`"first"`/`"last"` an end. The default is each midplane, or for a 2-D field
the whole plane.

#### `glyphs(*, components="cartesian", stride=2, scale=None, cmap="viridis", show_domain=True, title=None, plotter=None, **selection)`
Arrows of a `(component, e1, e2, e3)` vector field, colored by magnitude.
`components="contravariant"` pushes logical components forward first.

#### `streamlines(*, components="cartesian", n_points=100, source_radius=None, source_center=None, max_length=None, tube_radius=None, cmap="viridis", show_domain=True, title=None, plotter=None, **selection)`
Field lines traced both ways from seeds in a sphere. For a 2-D field, seeded
on the plane and kept on it.

#### `movie(path, *, kind="slices", step=1, framerate=10, clim=None, **options)`
One view per time step (`kind` is `"isosurface"`, `"slices"`, `"glyphs"` or
`"streamlines"`) into a `.gif` (needs `imageio`) or video (needs
`imageio-ffmpeg`), with shared color limits and a fixed camera. `options` go
to the view.

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

### `orbit_classification(*, x="v_par", y=None, v_par="v_par", t="first", ax=None, s=8)`
For a guiding-center orbits product (Particles5D or Particles5Dvperp): scatter
markers in a phase-space plane, colored as passing, trapped or lost, with each
class's count and fraction in the legend. The classification follows Struphy's
criteria (see `classify_orbits()` in the analysis reference). `y` defaults to
`mu`, or `v_perp` if there is no `mu`; `x="p_phi"` gives the canonical-momentum
diagram when `p_phi` was saved. `t` selects the time plotted (default: the
initial positions). `result.data["counts"]` holds the counts, and
`dataset.struphy.data.orbit_classification(...)` returns the plotted data.

### `orbits_3d(*, color_by="t", max_markers=200, tube_radius=None, cmap=None, domain=None, title=None, plotter=None)`
PyVista orbit lines (or tubes) from the physical positions `x`, `y`, `z`,
colored by `"t"`, `"classification"` (a legend of passing, trapped, lost) or
any `(t, marker)` variable. Samples where a marker is lost are dropped.
`domain` is a field whose boundary is drawn for context.

## Standalone functions

`struphy_plots.plotting` also exposes functions that don't hang off a single
array or dataset:

### `plot_convergence(sizes, errors, *, ax=None, order=None, label=None, xlabel="resolution", title="Convergence")`
Log-log plot of an error norm against resolution or step size. With
`order=None` (default), fits and draws the observed order via
`struphy_plots.analysis.convergence_order`; pass an explicit `order` to draw
a reference slope instead.

### `plot_continuous_spectrum(spectrum, x, modes, *, frequencies=None, mode_label="(m, n)", xlabel="x", ax=None, title="Continuous spectrum")`
Continuum frequencies `omega(x)` for each mode, with one color per mode and
one line style per branch. `spectrum` is called as `spectrum(x, *mode)` and
returns a mapping of branch name to `omega(x)`, for example Struphy's
`MhdContinousSpectraShearedSlab` or `MhdContinousSpectraCylinder` (shear
Alfvén and slow sound continua) with `modes=[(1, -1), (2, -1)]`.
`frequencies` marks measured frequencies as horizontal lines, to check
whether a mode sits in a continuum gap or crosses a continuum (where it is
damped):

```python
from struphy.dispersion_relations.analytic import MhdContinousSpectraShearedSlab
from struphy_plots.plotting import plot_continuous_spectrum

plot_continuous_spectrum(
    MhdContinousSpectraShearedSlab(),
    np.linspace(0, 1, 200),
    [(m, -1) for m in range(1, 4)],
    frequencies={"measured": 0.12},
)
```

`prepare_continuous_spectrum(spectrum, x, modes)` returns the evaluated
curves as a `(mode, branch, x)` `xarray.DataArray`, without plotting them.
