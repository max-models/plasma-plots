# struphy-plots

Optional plotting and diagnostics layer for [Struphy](https://github.com/struphy-hub/struphy)
output, extracted into its own package so it can evolve and release
independently of the Struphy runtime.

Full documentation: https://struphy-hub.github.io/struphy-plots

For development, install it with `pip install -e .`.
Import `struphy_plots` after installing it to register the optional
`xarray.DataArray.struphy` accessor on Struphy output arrays. For example,
`out.evaluate("em_fields/phi").struphy.plot.slice(x="eta1", y="eta2", t="last")`.
Direct plotting functions are available from
`struphy_plots.plotting`; analysis functions are in `struphy_plots.analysis`.

The accessor provides time-series, lineout, slice, panel, vector, comparison,
animation, and marker-trajectory plots. For three-dimensional scalar fields,
install the optional PyVista dependency (`pip install struphy-plots[pyvista]`) and
use `field.struphy.plot.volume(t=-1)`, then call `show()` on the returned plotter.
