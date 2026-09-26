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

- **`energies(parts=None, total="en_tot", groups=None)`** — the run's energy
  budget: its `en_*` scalars, the relative drift of the total (which should
  stay flat), and with `groups` the energy exchanged between them. For a run
  with energetic ions driving a wave:

  ```python
  out.plot.energies(groups={"wave": ["en_U", "en_B", "en_p"], "energetic ions": ["en_fv", "en_fB"]})
  ```

  Where energy only moves between the two groups, the wave's gain and minus
  the ions' change (dashed) overlap.

  ![Energy parts, total-energy drift and energy exchange between wave and energetic ions](../../../assets/figures/energy_budget.png)

- **`equilibrium(ax=None)`** — radial profiles of the run's fluid
  equilibrium (`out.equil`, `out.domain`): pressure, and density/temperature
  if the equilibrium has a density profile too.

  ![Radial equilibrium profiles: pressure, density, temperature](../../../assets/figures/equilibrium.png)

- **`equilibrium_3d(scalars="p0", cmap="viridis")`** — interactive 3-D
  equilibrium view via PyVista. Requires
  `pip install "struphy-plots[pyvista]"`.

  ![A 3-D equilibrium view shaded by a scalar field](../../../assets/figures/equilibrium_3d.png)

- **`domain_3d(n1=8, n2=32, n3=32, surface=True)`** — a PyVista wireframe of
  the run's mapping (`out.domain`), for checking its geometry. See
  [3-D views](/struphy-plots/guides/3d-views/#the-domain).

  ![Wireframe of a toroidal mapping](../../../assets/figures/3d_domain.png)

See the [Whole-run reference](/struphy-plots/reference/output/) for full
signatures.

## Energies from fields

`out.analysis.linear_mhd_energies()` recomputes LinearMHD's energy scalars
(`en_U`, `en_B`, `en_thermal`, `en_p`, `en_tot`) from the saved fields. It uses
the run's mapping and equilibrium, at the Gauss points of every element. On
real runs it matches the scalars Struphy saves during the simulation to machine
precision. Its purpose is to measure the energy of fields that were never
simulated, such as a filtered mode:

```python
etas, weights = out.analysis.quadrature_grid()        # Gauss points and weights per direction
u = out.evaluate("mhd/velocity", eta1=etas["e1"], eta2=etas["e2"], eta3=etas["e3"], representation="2")
mode = u.struphy.analysis.filter_time(pad_bins=1).filtered
out.analysis.linear_mhd_energies(velocity=mode, b_field=None, pressure=None).en_U   # energy in that mode
```

Fields must be in their FEEC space's own representation: 2-form components
for velocity and magnetic field, a 3-form for pressure. The default
post-processing products use other representations (`"norm"`, `"0"`). For
other models, `struphy_plots.analysis.field_energy` and `volume_integral`
compute the same kinds of integrals for any form. See the
[Diagnostics guide](/struphy-plots/guides/analysis/#volume-integrals-and-field-energies).

