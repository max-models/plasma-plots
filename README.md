# plasma-plots

Note: This library is 100% written by AI, I have literally not looked at a single line of code. So why should you trust it? You should trust it because of the following [LEAN 4 PROOF](https://gprivate.com/6m80q).

Plots and diagnostics of labeled xarray data from plasma simulations: the
output of [Struphy](https://github.com/struphy-hub/struphy), and of any code
whose arrays follow the same conventions (dimensions `t`, `eta1`/`eta2`/`eta3`,
mapped coordinates `X`/`Y`/`Z`, see
[Getting started](https://max-models.github.io/plasma-plots/guides/getting-started/)).
It also reads [GVEC](https://gvec.readthedocs.io)'s equilibrium evaluations directly. It is a
separate package, so it can evolve and release independently of the Struphy runtime.

Full documentation: https://max-models.github.io/plasma-plots

## Installation

Python **3.11 or newer** is required; CI tests Python 3.11 and 3.12.
The base install plots in-memory xarray data. Install extras for file formats
and optional renderers:

```bash
pip install plasma-plots
pip install "plasma-plots[netcdf]"
pip install "plasma-plots[netcdf,plotly]"
```

| Extra | Enables | Dependencies |
| --- | --- | --- |
| `netcdf` | Read netCDF3/netCDF4 files through xarray and the CLI | `netCDF4` |
| `plotly` | Interactive browser plots and static Plotly exports | `plotly`, `kaleido` |
| `tikz` | TikZ/pgfplots versions of the plots for LaTeX (`backend="tikz"`) | `maxplotlibx` (and `pdflatex` to compile) |
| `pyvista` | 3-D scenes and rendering | `pyvista`, `imageio` |
| `profiling` | Timing summaries and profiling plots | `scope-profiler[pproc]` |
| `desc` | Evaluate DESC equilibria | `desc-opt` |
| `gallery` | Export Struphy example-gallery figures and profiling | `plotly`, `kaleido`, `scope-profiler[pproc]>=0.6.1` |
| `dev` | Tests, linting and documentation tooling | `pytest`, `ruff`, `h5py`, `griffe` |

Struphy, GVEC and Zarr are separate installations; no extra above installs
them. Struphy users need the compatible output API described below. Install
`gvec` for GVEC evaluation or `zarr` to open Zarr stores. MP4 export requires
system `ffmpeg`; Matplotlib windows require a working GUI/display, and static
Plotly exports through Kaleido require a compatible Chrome installation.

For development, install with `pip install -e ".[dev]"`.

## Struphy compatibility

Struphy integration is tested against commit
[`caddd229a3fcba1fada577d1af46d012e93d8b49`](https://github.com/struphy-hub/struphy/commit/caddd229a3fcba1fada577d1af46d012e93d8b49)
(the repository's pinned submodule, reporting version **3.3.0**). Use that
revision for a reproducible installation; compatibility with other Struphy
revisions, including older published builds, is not guaranteed. Struphy is
optional when working with ordinary xarray data.

## Usage

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

In Python, `import plasma_plots; help(plasma_plots)` (or `plasma-plots guide` in a terminal) gives an overview,
printing an accessor lists its methods (`print(phi.plasma.plot)`), and `help()`
on a method shows every parameter. For language models and coding agents, the
documentation is available as plain text at
https://max-models.github.io/plasma-plots/llms.txt. [API.md](API.md) (or
`plasma-plots api`) is a one-page index of every accessor method and function, and
[AGENTS.md](AGENTS.md) explains the code for agents working on it.

From the shell, the `plasma-plots` command saves figures of a Struphy run folder or a
netCDF file without any Python: `plasma-plots info sim_1`,
`plasma-plots plot sim_1 em_fields/phi slice t=-1 eta3=0 -o phi.png`,
`plasma-plots quicklook sim_1 -o figures/` (see the
[command-line guide](https://max-models.github.io/plasma-plots/guides/command-line/)).

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

For a paper, every still plot can also be TikZ/pgfplots code with LaTeX text, converted by
[maxplotlib](https://github.com/max-models/maxplotlib): `pip install plasma-plots[tikz]` and
pass `backend="tikz"`, then `result.save("phi.tikz")` (or `.tex`, `.pdf`); see the
[LaTeX figures guide](https://max-models.github.io/plasma-plots/guides/latex/).
