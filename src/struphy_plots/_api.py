"""A compact index of the whole API, for people and coding agents: ``python -m struphy_plots api``.

One line per accessor method and public function, with its signature and the first line of its
docstring, grouped by where it is reached (``array.struphy.plot``, ``out.plot``, ...), after the
conventions every method shares. It is generated from the code, so it cannot go stale; ``API.md``
in the repository and ``/llms-api.txt`` on the docs site are this text (``scripts/api_index.py``).
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import re

from ._docs import plain, summary

HEADER = """# struphy-plots API index

Plots and diagnostics of labeled xarray data from plasma simulations (Struphy, and any code
whose output follows the same conventions, e.g. GENE). Generated from the code by
`python -m struphy_plots api`: every accessor method and public function, with its signature
and one-line summary. `help()` on any of them shows every parameter; the guides with figures are
at https://struphy-hub.github.io/struphy-plots.

## Conventions every method shares

- **Accessors, not functions.** `import struphy_plots` adds `.struphy` to every
  `xarray.DataArray` and `xarray.Dataset`: `array.struphy.plot.*`, `.analysis.*`, `.data.*`. A
  Struphy `Output` also gets `out.plot.*` and `out.analysis.*` (and loads struphy-plots itself).
- **Selection by keyword.** Name every dimension a plot does not draw: an integer is a position
  (`t=-1` the last, `t=0` the first), a float the nearest coordinate value (`t=0.35`).
  Unknown names, strings and bools raise `TypeError`.
- **Dimensions and attributes.** Time is `t`; logical space `eta1`, `eta2`, `eta3` (mapped
  coordinates `X`, `Y`, `Z` as 2-D/3-D coordinates); vector components `component`; markers
  `marker`. Labels come from `attrs["label"]` (mathtext), units from `attrs["units"]`,
  coordinate labels from their `long_name`/`units`; `attrs["run"]` titles a figure,
  `attrs["run_name"]` names a run in comparisons.
- **Return values.** Plots return a `PlotResult` (`.fig`, `.ax`, `.artists`, `.fit_results`,
  `.data`, `.save(path)`, `.show()`, `.to_plotly()`); Matplotlib animations a `FuncAnimation`;
  PyVista views a `pyvista.Plotter`; analysis methods labeled xarray objects (which have
  `.struphy` again) or small result dataclasses (`FitResult`, `OscillationFit`, ...).
  `array.struphy.data.<plot>(...)` returns the data a plot would draw, without drawing it.
- **Backends.** Every Matplotlib plot takes `backend="plotly"` for an interactive Plotly figure
  (in a `PlotResult`; animations get a slider); `struphy_plots.set_backend("plotly")` sets the
  default. `ax=` draws into your own axes (Matplotlib only); `struphy_plots.figure(rows, cols)`
  composes several plots into one figure, with either backend.
- **MPI.** Under `mpirun`, plots draw and save on rank 0 only; other ranks get a `SkippedPlot`
  whose methods do nothing. Analysis runs on every rank.
- **Theory.** `struphy_plots.theory.*` are plain functions (complex ω for dispersion relations)
  that go straight into `branches=`, `reference=` and `theory=` of the plots.

## Parameters that mean the same everywhere

- **2-D slices** (`slice`, `panels`, `animation`, `viewer`, `frames`, `view`) draw the two
  dimensions `x=` and `y=` (logical, e.g. `x="eta1", y="eta2"`, or `x="eta1", y="t"` for a
  space-time map), or with `coords="physical", plane="XY"` the mapped coordinates; `plane` is
  one of `"XY"`, `"XZ"`, `"YZ"`, `"RZ"` (`"RZ"`: R = √(X² + Y²), the poloidal plane of a torus).
  `symmetric=True` centres the colours on zero, `robust=True` clips outliers, `levels=` adds
  contour lines, `overlays={...}` a second field's contours, the boundary, lines and points;
  `xlabel=`, `ylabel=`, `colorbar_label=` replace the labels.
- **Fits on plots** take a time window, not a fit: `plot.timeseries(fit=(t0, t1))` (or `fit=True`
  for all), `fit_amplitude=True` for a squared quantity such as an energy; the fit is then in
  `result.fit_results[0]`. **Fits in analysis** take `window=(t0, t1)`:
  `analysis.growth_rate(window=(t0, t1)).rate`.
- **References and theory:** `reference=` (a function of `t` or `x`, or of `x` and `t`, an array,
  an `(x, y)` pair, or a dict of labels to these), `branches={"label": omega_of_k}`,
  `theory=` of `against_theory`.
- **Animations:** `step=n` keeps every n-th frame, `max_frames=n` at most n evenly spaced ones;
  `interval=` is the delay in ms. With `backend="plotly"` they return a `PlotResult` with a slider;
  `result.save("movie.png", frame=k)` saves frame k as the image.
- **Markers** (`dataset.struphy.plot.*` of orbits or particles): `x=`/`y=` name variables of the
  Dataset (`x="R"` is √(x² + y²) if not a variable), `color=` a variable name or
  `"classification"` (passing/trapped/lost, needs `v_par`); animations take `trail=n`,
  `paths=True`, `background=` a field.
- **Composing:** `with struphy_plots.figure(2, 1, sharex=True) as fig:` then `ax=fig[0]`,
  `ax=fig[1]` (flat index, or `fig[row, col]`); `fig.save(...)` after the block (inside it,
  it saves the panels drawn so far).
"""

# (heading, how it is reached, class path)
ACCESSORS = [
    ("array.struphy.plot", "array.struphy.plot", "accessors.ArrayPlots"),
    ("array.struphy.plot.view(...)", "view", "accessors.SliceView"),
    ("array.struphy.analysis", "array.struphy.analysis", "accessors.ArrayAnalysis"),
    ("array.struphy.data", "array.struphy.data", "accessors.ArrayData"),
    ("dataset.struphy.plot", "dataset.struphy.plot", "accessors.DatasetPlots"),
    ("dataset.struphy.analysis", "dataset.struphy.analysis", "accessors.DatasetAnalysis"),
    ("dataset.struphy.data", "dataset.struphy.data", "accessors.DatasetData"),
    ("out.plot (a Struphy Output)", "out.plot", "output_accessors.OutputPlots"),
    ("out.plot.profile", "out.plot.profile", "output_accessors.ProfilePlots"),
    ("out.analysis (a Struphy Output)", "out.analysis", "output_accessors.OutputAnalysis"),
]
TOP_LEVEL = [
    "struphy_plots.figure",
    "struphy_plots.set_backend",
    "struphy_plots.get_backend",
    "struphy_plots.mpi_rank",
    "struphy_plots.is_plotting_rank",
    "struphy_plots.plotting.PlotResult.save",
    "struphy_plots.plotting.PlotResult.show",
    "struphy_plots.plotting.PlotResult.to_plotly",
    "struphy_plots.plotly_backend.to_plotly",
    "struphy_plots.plotly_backend.animation_to_plotly",
]
RESULTS = [
    "struphy_plots.plotting.PlotResult",
    "struphy_plots.analysis.FitResult",
    "struphy_plots.analysis.OscillationFit",
    "struphy_plots.analysis.ConvergenceFit",
    "struphy_plots.analysis.BranchFit",
    "struphy_plots.spectral.TimeFilterResult",
    "struphy_plots.analysis.GrowthFit",
    "struphy_plots.plotting.View",
]
# the modules of plain functions behind the accessors, for arrays from anywhere
MODULES = ["plotting", "spectral_plots", "analysis", "spectral", "arrays", "pyvista_plots"]
THEORY = ["kinetic", "waves", "parameters", "orbits", "exact", "numerics", "special"]


def _default(value) -> str:
    if inspect.isfunction(value):
        return value.__name__
    text = repr(value)
    return text if len(text) <= 24 else text[:21] + "..."


def signature(function) -> str:
    """The parameters of ``function`` as compact text: names and defaults, no annotations."""
    try:
        parameters = inspect.signature(function).parameters.values()
    except (TypeError, ValueError):
        return "(...)"
    parts, keyword_only = [], False
    for p in parameters:
        if p.name in ("self", "cls"):
            continue
        if p.kind is p.VAR_POSITIONAL:
            parts.append(f"*{p.name}")
            keyword_only = True
            continue
        if p.kind is p.KEYWORD_ONLY and not keyword_only:
            parts.append("*")
            keyword_only = True
        if p.kind is p.VAR_KEYWORD:
            parts.append(f"**{p.name}")
        elif p.default is p.empty:
            parts.append(p.name)
        else:
            parts.append(f"{p.name}={_default(p.default)}")
    return "(" + ", ".join(parts) + ")"


def _line(name, function) -> str:
    return f"- `{name}{signature(function)}`: {summary(function)}"


def examples(function, limit=2) -> list[str]:
    """The first ``limit`` statements of the Examples section of a docstring."""
    doc = inspect.getdoc(function) or ""
    match = re.search(r"^Examples\n-+\n(.*?)(?=^\S[^\n]*\n-{3,}|\Z)", doc, re.S | re.M)
    if not match:
        return []
    found = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith(">>> "):
            code = stripped[4:]
            if code.startswith(("import ", "from ")) or len(found) >= limit:
                continue
            found.append(code)
        elif stripped.startswith("... ") and found:
            found[-1] += " " + stripped[4:].strip()
    return found


def _with_examples(name, function) -> list[str]:
    return [_line(name, function)] + [f"  - e.g. `{code}`" for code in examples(function)]


def _resolve(path):
    module, _, rest = path.partition(".")
    obj = importlib.import_module(f"struphy_plots.{module}") if module != "struphy_plots" else None
    if obj is None:
        import struphy_plots as obj
    for part in rest.split("."):
        obj = getattr(obj, part)
    return obj


def _public_functions(module):
    return [
        (name, obj)
        for name, obj in vars(module).items()
        if not name.startswith("_") and inspect.isfunction(obj) and obj.__module__ == module.__name__
    ]


def api_index() -> str:
    """The API index as Markdown: the shared conventions, then one line per method and function.

    Returns
    -------
    str
        The index; ``python -m struphy_plots api`` prints it.
    """
    import struphy_plots  # noqa: F401  (registers the accessors)

    out = [HEADER]
    out.append("## Accessor methods\n")
    for heading, reached, path in ACCESSORS:
        cls = _resolve(path)
        doc = plain(inspect.getdoc(cls) or "").split("\n\n")[0].replace("\n", " ")
        out.append(f"### {heading}\n\n{doc}\n")
        for name, member in vars(cls).items():
            if name.startswith("_") and name != "__call__":
                continue
            if isinstance(member, property):
                out.append(f"- `{reached}.{name}`: {summary(member.fget)}")
            elif inspect.isfunction(member):
                shown = reached if name == "__call__" else f"{reached}.{name}"
                out += _with_examples(shown, member)
        out.append("")
    out.append("## Top level\n")
    for path in TOP_LEVEL:
        out += _with_examples(path, _resolve(path))
    out.append("")
    out.append("## Result types\n")
    out.append("What the methods return, with their fields (`help()` on the class explains each).\n")
    for path in RESULTS:
        cls = _resolve(path)
        fields = [f.name for f in dataclasses.fields(cls) if not f.name.startswith("_")]
        out.append(f"- `{cls.__name__}({', '.join(fields)})`: {summary(cls)}")
    out.append("")
    out.append("## Plain functions behind the accessors\n")
    out.append(
        "The same plots and diagnostics as functions of arrays, e.g. `plot_slice(phi.isel(t=-1))`; each "
        "accessor method names its function first under See Also, and `help()` shows the parameters.\n"
    )
    for name in MODULES:
        module = importlib.import_module(f"struphy_plots.{name}")
        names = [fname for fname, _ in _public_functions(module)]
        if names:
            out.append(f"- `struphy_plots.{name}`: " + ", ".join(f"`{n}`" for n in names))
    out.append("")
    out.append("## Theory: struphy_plots.theory\n")
    out.append("Analytic results to compare with, plain numpy; complex ω for dispersion relations.\n")
    for name in THEORY:
        module = importlib.import_module(f"struphy_plots.theory.{name}")
        out.append(f"### struphy_plots.theory.{name}\n")
        out += [_line(fname, function) for fname, function in _public_functions(module)]
        out.append("")
    return "\n".join(out).rstrip() + "\n"
