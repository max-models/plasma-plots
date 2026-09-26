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
guide, whose `real_*.png`/`real_plotly_*.json` figures come from actually
running struphy (`scripts/generate_real_example_figures.py`), to show the
same functions working end-to-end on genuine output. That script needs the
full **compiled** `struphy` runtime (from the submodule, see above --
`struphy compile -y` if you haven't).

Figures aren't checked into git, so generate them before building or
running the site locally:

```bash
struphy compile -y   # only needed for generate_real_example_figures.py
pip install -e ".[pyvista,profiling]" plotly   # see below for why these extras
make figures       # runs both scripts, renders docs/src/assets/figures/ and docs/public/plotly/
make docs-dev       # figures + npm run dev
make docs-build     # figures + npm run build
```

`generate_real_example_figures.py` takes a couple of minutes (it's running
several real simulations, including a kinetic/PIC one) -- this is expected,
not a bug.

- Without the `pyvista` extra (or without a working display), each script
  skips its PyVista-based figures (`volume.png`/`real_volume.png`,
  `equilibrium_3d.png`/`real_equilibrium_3d.png`) and prints a warning --
  but the docs pages that embed them will then fail to build, since Astro
  needs the referenced file to exist. CI runs figure generation under
  `xvfb-run` with Mesa's software renderer for exactly this reason (see
  `.github/workflows/docs.yml`); do the same locally if you hit rendering
  errors.
- Without the `profiling` extra, `generate_docs_figures.py` skips its three
  `profile_*.png` figures and `plotly_profile_gantt.json` the same way, and
  `generate_real_example_figures.py` skips the `real_` equivalents (built
  from actually profiling the kinetic run, via `sim.run(profiling_activated=True)`).
- `plotly` exports each interactive figure's JSON, fetched client-side by
  the `<PlotlyChart>` component (`docs/src/components/PlotlyChart.astro`),
  which loads Plotly.js itself from a CDN at view time -- neither is a
  `struphy-plots` dependency, and the docs pages that use `<PlotlyChart>`
  will fail to build without the referenced JSON, the same as the PyVista
  figures above.

CI regenerates the figures the same way (with the same heavy struphy install
as the test suite, since the real-example page still needs it) on every push
to `devel`, before deploying to GitHub Pages.

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
