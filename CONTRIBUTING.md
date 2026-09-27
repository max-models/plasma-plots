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

The test suite exercises `struphy_plots` against real `struphy` output
objects (`struphy.post_processing.output.Output` and friends, plus `h5py`),
built from small fixtures that don't need compiled kernels, so `struphy compile`
isn't required just to run `pytest`.

## Docs

The documentation site (Astro + Starlight) lives in `docs/`. Almost every
figure and interactive Plotly chart (`docs/src/assets/figures/*`,
`docs/public/plotly/*.json`) is built from small **synthetic** `xarray` data
by `scripts/generate_docs_figures.py` -- the point of those guides is
`struphy_plots` itself, not any particular physics, so this needs only
`struphy-plots` and its optional extras, no struphy install at all.

The one exception is the [A real
simulation](https://struphy-hub.github.io/struphy-plots/guides/real-example/)
guide, whose `real_*.png` figures (and the full source shown on that page,
via an Astro `?raw` import) come from actually running struphy
(`scripts/generate_real_example_figures.py`, a real `LinearMHD` slab-waves
run), to show the same functions working end-to-end on genuine output. That
script needs the full **compiled** `struphy` runtime (from the submodule,
see above -- `struphy compile -y` if you haven't); `scipy` comes along with
it (struphy's own dependency).

Figures aren't checked into git, so generate them before building or
running the site locally:

```bash
struphy compile -y   # only needed for generate_real_example_figures.py
pip install -e ".[pyvista,profiling]" plotly   # see below for why these extras
make figures       # runs both scripts, renders docs/src/assets/figures/ and docs/public/plotly/
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
  `struphy-plots` dependency, and the docs pages that use `<PlotlyChart>`
  will fail to build without the referenced JSON, the same as the PyVista
  figures above.

CI regenerates the figures the same way (with the same heavy struphy install
as the test suite, since the real-example page still needs it) on every push
to `devel`, before deploying to GitHub Pages.

## Docstrings and the API reference

The API reference on the docs site is generated from the docstrings by
[starlight-pydocs](https://github.com/ewels/starlight-pydocs) when the site builds, so it never
drifts from the code. It runs griffe (through `uv` if it's on your `PATH`, else `python -m griffe`,
which the `dev` extra installs), with the extension `scripts/griffe_extension.py`: that fills in
inherited parameters (see below) and turns the Sphinx roles into links. The module pages live
under `/api/struphy_plots/`. The accessor pages (`docs/src/content/docs/reference/plot.mdx` and
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
- **Accessor methods** (`array.struphy.plot.slice`, ...) that wrap a function
  name it first under `See Also`. Parameters the method doesn't document are
  then taken from that function, so the two can't disagree. Document a
  parameter in the method only when it differs from the function's
  (`**selection`, for example). The first entry may also be another method
  with the same options: `slice`, `panels`, `viewer`, `animation` and
  `frames` all take theirs from `ArrayPlots.view`.
- **Markup:** ``` ``code`` ```, cross-references as `` :func:`struphy_plots.spectral.fft` ``,
  `` :meth:`ArrayPlots.slice` `` or `` :class:`PlotResult` `` (a leading `~`
  shows only the last name), and Unicode instead of LaTeX (`ω`, `|√g|`).
- **Selections** in examples use integer positions (`t=-1`, `t=0`) or float
  values (`t=0.35`).

Code in `struphy_plots.theory` is plain numpy (no scipy, no Struphy), follows the conventions
in its package docstring (complex frequencies, branch dicts, normalized units), cites its
sources under `References`, and is tested against literature values or independent numerical
solutions. Its docstring examples must run: `tests/test_theory_doctests.py` runs them.

`python scripts/check_docstrings.py` lists every docstring that breaks these rules, and
`tests/test_docstrings.py` runs the same check in CI, together with a check that the reference
shows every parameter of every function.

## Branches and releases

- **`devel`** — default development branch. Every push rebuilds and deploys
  the docs site to GitHub Pages (`.github/workflows/docs.yml`).
- **`main`** — release branch. Every push builds the package and publishes it
  to PyPI (`.github/workflows/publish.yml`), so bump `version` in
  `pyproject.toml` before merging `devel` into `main`. Publishing uses
  [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC),
  so no API token is stored in the repo — this needs a one-time "pending
  publisher" set up on the PyPI project settings for
  `struphy-hub/struphy-plots`, workflow `publish.yml`, environment `pypi`.
  The workflow passes `skip-existing: true`, so re-pushing `main` without a
  version bump is a safe no-op rather than a failing build.
