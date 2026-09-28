# plasma-plots

Note: This library is 100% written by AI, I have literally not looked at a single line of code. So why should you trust it? You should trust it because of the following [LEAN 4 PROOF](https://gprivate.com/6m80q).

Plots and diagnostics of labeled xarray data from plasma simulations: the
output of [Struphy](https://github.com/struphy-hub/struphy), and of any code
whose arrays follow the same conventions (dimensions `t`, `eta1`/`eta2`/`eta3`,
mapped coordinates `X`/`Y`/`Z`, see
[Getting started](https://struphy-hub.github.io/plasma-plots/guides/getting-started/)).
It also reads [GVEC](https://gvec.readthedocs.io)'s equilibrium evaluations directly. It is a
separate package, so it can evolve and release independently of the Struphy runtime.

Full documentation: https://struphy-hub.github.io/plasma-plots

For development, install it with `pip install -e .`.

Creating a Struphy `Output` loads plasma-plots, which registers `out.plot`,
`out.analysis` and the `.plasma` accessor on every product:

```python
from struphy.post_processing.output import Output

out = Output("path/to/run")
out.evaluate("em_fields/phi").plasma.plot.slice(x="eta1", y="eta2", t=-1)
```

> [!IMPORTANT]
> For xarray data that doesn't come from an `Output`, `import plasma_plots`
> first. Without it you get
> `AttributeError: 'DataArray' object has no attribute 'plasma'`.

In Python, `import plasma_plots; help(plasma_plots)` (or `python -m plasma_plots`) gives an overview,
printing an accessor lists its methods (`print(phi.plasma.plot)`), and `help()`
on a method shows every parameter. For language models and coding agents, the
documentation is available as plain text at
https://struphy-hub.github.io/plasma-plots/llms.txt. [API.md](API.md) (or
`python -m plasma_plots api`) is a one-page index of every accessor method and function, and
[AGENTS.md](AGENTS.md) explains the code for agents working on it.

Direct plotting functions are available from
`plasma_plots.plotting`; analysis functions are in `plasma_plots.analysis`.

The accessor provides time-series, lineout, slice, panel, vector, comparison,
animation, and marker-trajectory plots. For three-dimensional scalar fields,
install the optional PyVista dependency (`pip install plasma-plots[pyvista]`) and
use `field.plasma.plot.volume(t=-1)`, then call `show()` on the returned plotter.

Every Matplotlib plot can also be an interactive Plotly figure, with hover values, zoom and,
for animations, a slider: `pip install plasma-plots[plotly]` and pass `backend="plotly"`
(`field.plasma.plot.slice(t=-1, backend="plotly")`), or call
`plasma_plots.set_backend("plotly")` once.
