# struphy-plots

Optional plotting and diagnostics layer for [Struphy](https://github.com/struphy-hub/struphy)
output, extracted into its own package so it can evolve and release
independently of the Struphy runtime.

Full documentation: https://struphy-hub.github.io/struphy-plots

For development, install it with `pip install -e .`.

> [!IMPORTANT]
> Always `import struphy_plots` first. The import registers the `.struphy`
> accessor on every `xarray.DataArray` and `Dataset` (and `out.plot` on
> Struphy's `Output`). Recent Struphy versions load it when an `Output` is
> created; older ones and plain xarray data don't. Without it you get
> `AttributeError: 'DataArray' object has no attribute 'struphy'`.

```python
import struphy_plots  # required: registers DataArray.struphy
from struphy.post_processing.output import Output

out = Output("path/to/run")
out.evaluate("em_fields/phi").struphy.plot.slice(x="eta1", y="eta2", t=-1)
```

In Python, `help(struphy_plots)` (or `python -m struphy_plots`) gives an overview,
printing an accessor lists its methods (`print(phi.struphy.plot)`), and `help()`
on a method shows every parameter. For language models and coding agents, the
documentation is available as plain text at
https://struphy-hub.github.io/struphy-plots/llms.txt.

Direct plotting functions are available from
`struphy_plots.plotting`; analysis functions are in `struphy_plots.analysis`.

The accessor provides time-series, lineout, slice, panel, vector, comparison,
animation, and marker-trajectory plots. For three-dimensional scalar fields,
install the optional PyVista dependency (`pip install struphy-plots[pyvista]`) and
use `field.struphy.plot.volume(t=-1)`, then call `show()` on the returned plotter.
