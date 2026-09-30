# Contributing

## Local development

`struphy` lives in this repo as a git submodule (`struphy/`), pinned to a
known-good commit -- clone/update it, then install from it:

```bash
git submodule update --init --recursive
pip install -e ./struphy   # struphy itself; needs a Fortran/C compiler + MPI + netCDF
pip install -e ".[dev,profiling]"
pytest
ruff check src tests scripts
```

The test suite exercises `plasma_plots` against real `struphy` output
objects (`struphy.post_processing.output.Output` and friends, plus `h5py`),
built from small fixtures that don't need compiled kernels, so `struphy compile`
isn't required just to run `pytest`.

## Docs

The documentation site (Astro + Starlight) lives in `docs/`. Almost every
figure and interactive Plotly chart (`docs/src/assets/figures/*`,
`docs/public/plotly/*.json`) is built from small **synthetic** `xarray` data
by `scripts/generate_docs_figures.py` -- the point of those guides is
`plasma_plots` itself, not any particular physics, so this needs only
`plasma-plots` and its optional extras, no struphy install at all.

The one exception is the [A real
simulation](https://struphy-hub.github.io/plasma-plots/guides/real-example/)
guide, whose `real_*.png` figures (and the full source shown on that page,
via an Astro `?raw` import) come from actually running struphy
(`scripts/generate_real_example_figures.py`, a real `LinearMHD` slab-waves
run), to show the same functions working end-to-end on genuine output. That
script needs the full **compiled** `struphy` runtime (from the submodule,
see above -- `struphy compile -y` if you haven't); `scipy` comes along with
it (struphy's own dependency).

The [GVEC equilibria](https://struphy-hub.github.io/plasma-plots/guides/gvec/) guide is the
other: its `gvec_*` figures, interactive 3-D scenes and `docs/src/assets/gvec/numbers.txt` come
from real GVEC equilibria (`scripts/generate_gvec_figures.py`: GVEC's tutorial stellarator and
tokamak, and W7-X), so it needs `pip install gvec`, which builds GVEC's Fortran core (gfortran, a
LAPACK and CMake). W7-X takes about a minute; `PLASMA_PLOTS_SKIP_W7X=1` leaves it out, but then
the page misses its figures.

The [DESC equilibria](https://struphy-hub.github.io/plasma-plots/guides/desc/) guide evaluates the
example equilibria that DESC ships (`scripts/generate_desc_figures.py`: W7-X, the precise QA and QH
stellarators, NCSX, HELIOTRON, ESTELL and the tokamak DSHAPE), so it needs `pip install desc-opt`
(pure Python, on JAX; the `desc` extra). It takes about two minutes. `tests/test_desc.py` needs it
too, and is skipped without it.

Figures aren't checked into git, so generate them before building or
running the site locally:

```bash
struphy compile -y   # only needed for generate_real_example_figures.py
pip install gvec     # only needed for generate_gvec_figures.py
pip install desc-opt # only needed for generate_desc_figures.py (and tests/test_desc.py)
pip install -e ".[pyvista,profiling]" plotly   # see below for why these extras
make figures       # runs the four scripts, renders docs/src/assets/figures/ and docs/public/plotly/
make docs-dev       # figures + npm run dev
make docs-build     # figures + npm run build
```

`generate_real_example_figures.py` takes under a minute.

- Without the `pyvista` extra (or without a working display), `generate_docs_figures.py`
  skips its two PyVista-based figures (`volume.png`, `equilibrium_3d.png`) and prints a
  warning -- but the docs pages that embed them will then fail to build, since Astro needs
  the referenced file to exist. CI runs figure generation under `xvfb-run` with Mesa's
  software renderer for exactly this reason (see `.github/workflows/docs.yml`); do the same
  locally if you hit rendering errors.
- Without the `profiling` extra, `generate_docs_figures.py` skips its three `profile_*.png`
  figures and `plotly_profile_gantt.json` the same way (a synthetic `scope-profiler`
  session, unrelated to the real-example script, which doesn't profile anything).
- `plotly` exports each interactive figure's JSON, fetched client-side by
  the `<PlotlyChart>` component (`docs/src/components/PlotlyChart.astro`),
  which loads Plotly.js itself from a CDN at view time -- neither is a
  `plasma-plots` dependency, and the docs pages that use `<PlotlyChart>`
  will fail to build without the referenced JSON, the same as the PyVista
  figures above.

CI regenerates the figures the same way (with the same heavy struphy install
as the test suite, since the real-example page still needs it) on every push
to `devel`, before deploying to GitHub Pages.

The docs site shows about 80 characters of code before a block scrolls sideways, so the
` ```python ` blocks of the guides are formatted with `ruff format` at 79 characters (not the
package's 120): run `python scripts/format_docs_snippets.py` after editing them, and
`tests/test_docs_snippets.py` fails until you do. A trailing comment that doesn't fit its line
moves above the statement. The scripts included with `<Code>` aren't covered.

## Docstrings and the API reference

The API reference on the docs site is generated from the docstrings by
[starlight-pydocs](https://github.com/ewels/starlight-pydocs) when the site builds, so it never
drifts from the code. It runs griffe (through `uv` if it's on your `PATH`, else `python -m griffe`,
which the `dev` extra installs), with the extension `scripts/griffe_extension.py`: that fills in
inherited parameters (see below) and turns the Sphinx roles into links. The module pages live
under `/api/plasma_plots/`. The accessor pages (`docs/src/content/docs/reference/plot.mdx` and
the others) embed one method per `<Autodoc>` block, in source order; `scripts/accessor_pages.py`
writes them, so run it after adding or removing an accessor method (`tests/test_accessor_pages.py`
fails until you do).

Every public module, class, function and method (a name without a leading
underscore) needs a [NumPy-style](https://numpydoc.readthedocs.io/en/latest/format.html)
docstring:

```python
def plot_lineout(data, *, x=None, reference=None, x_of=None, ax=None):
    """Plot a one-dimensional profile along one coordinate.

    Every other dimension must already be selected. Longer explanation, in
    paragraphs, goes here.

    Parameters
    ----------
    data : xarray.DataArray
        The profile, with ``x`` as its only dimension.
    x : str, optional
        The coordinate along the horizontal axis. Default: the only dimension.
    reference : callable, array, (x, y) pair or dict, optional
        Exact profiles, drawn dashed: a function of the plotted ``x`` (or of
        ``x`` and ``t``), or a dict of labels to these.
    **selection
        Dimensions to select first: an integer is a position (``t=-1`` the
        last), a float the nearest coordinate value.

    Returns
    -------
    PlotResult
        The figure, the axes and the drawn lines.

    Raises
    ------
    ValueError
        If more than one dimension remains.

    See Also
    --------
    plot_profiles : Several profiles in one axes.

    Examples
    --------
    >>> plot_lineout(phi.isel(t=-1, eta2=0, eta3=0), x="eta1")
    """
```

The rules:

- **Summary line:** one sentence that ends with a period and fits on the first
  line.
- **Parameters:** every parameter of the signature (except `self`), as
  `name : type` with an indented description. The type is written for
  readers (`float or (float, float), optional`), since it's what the reference
  shows. Document `*args` and `**kwargs` as `*others` and `**selection`, i.e.
  by their names with the stars.
- **Returns:** whenever the function returns a value.
- **Raises**, **See Also** and **Examples** where they help. Examples use
  `>>>` and don't have to run on their own (they may use a field `phi` or an
  `Output` called `out`).
- **Classes:** result dataclasses (`PlotResult`, `FitResult`, ...) list their
  fields under `Attributes`. Other classes that users construct document
  their constructor under `Parameters`.
- **Accessor methods** (`array.plasma.plot.slice`, ...) that wrap a function
  name it first under `See Also`. Parameters the method doesn't document are
  then taken from that function, so the two can't disagree. Document a
  parameter in the method only when it differs from the function's
  (`**selection`, for example). The first entry may also be another method
  with the same options: `slice`, `panels`, `viewer`, `animation` and
  `frames` all take theirs from `ArrayPlots.view`.
- **Markup:** ``` ``code`` ```, cross-references as `` :func:`plasma_plots.spectral.fft` ``,
  `` :meth:`ArrayPlots.slice` `` or `` :class:`PlotResult` `` (a leading `~`
  shows only the last name), and Unicode instead of LaTeX (`ω`, `|√g|`).
- **Selections** in examples use integer positions (`t=-1`, `t=0`) or float
  values (`t=0.35`).

Code in `plasma_plots.theory` is plain numpy (no scipy, no Struphy), follows the conventions
in its package docstring (complex frequencies, branch dicts, normalized units), cites its
sources under `References`, and is tested against literature values or independent numerical
solutions. Its docstring examples must run: `tests/test_theory_doctests.py` runs them.

`python scripts/check_docstrings.py` lists every docstring that breaks these rules, and
`tests/test_docstrings.py` runs the same check in CI, together with a check that the reference
shows every parameter of every function.

## Plotly versions of the plots

`backend="plotly"` (`src/plasma_plots/plotly_backend.py`) converts the finished Matplotlib
figure into Plotly; there is no second implementation of any plot. A new accessor plot method
that draws with Matplotlib gets the option with `@with_backend` and a `backend: Backend | None =
None` parameter, documented in the method itself (copy the entry of a neighbouring method);
`tests/test_plotly_backend.py` fails for a method without it, and converts every plot with
conversion warnings as errors. If a plot starts using a Matplotlib artist the converter does not
know (a `ConversionWarning` says which), teach `_FigureConverter` to convert it and add the plot
to the test's cases.

## Branches and releases

- **`devel`** — default development branch. Every push rebuilds and deploys
  the docs site to GitHub Pages (`.github/workflows/docs.yml`).
- **`main`** — release branch. Every push builds the package and publishes it
  to PyPI (`.github/workflows/publish.yml`), so bump `version` in
  `pyproject.toml` before merging `devel` into `main`. Publishing uses
  [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC),
  so no API token is stored in the repo — this needs a one-time "pending
  publisher" set up on the PyPI project settings for
  `struphy-hub/plasma-plots`, workflow `publish.yml`, environment `pypi`.
  The workflow passes `skip-existing: true`, so re-pushing `main` without a
  version bump is a safe no-op rather than a failing build.

## Releasing

Merging or pushing to `main` triggers `.github/workflows/publish.yml` and
publishes to **PyPI** using the `pypi` environment's Trusted Publisher.
It then publishes the corresponding **GitHub release** (`v<version>`) and
attaches the same wheel and source archive. An existing draft is completed;
an already published GitHub release is left unchanged.

Before merging the release PR, wait for the Python tests and docs build,
update `pyproject.toml`'s version, and add `releases/<version>.md` with features,
optional dependencies and the exact tested Struphy revision. A draft release
can be created in advance; it does not publish the package. The workflow
validates distribution metadata before publishing. Configure PyPI Trusted
Publishing for this repository, `publish.yml`, and the `pypi` environment.
A manual dispatch on a non-main branch does not create a GitHub release;
release dispatches should use `main`.
