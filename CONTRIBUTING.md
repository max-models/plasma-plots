# Contributing

## Local development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

The test suite exercises `struphy_plots` against real `struphy` output
objects, so `struphy` (plus `h5py`) must be importable — install it from the
main repo (`pip install -e /path/to/struphy` or
`pip install git+https://github.com/struphy-hub/struphy.git@postprocessing`).

## Docs

The documentation site (Astro + Starlight) lives in `docs/`. It embeds
example figures and one animation (`docs/src/assets/figures/*`) rendered by
`scripts/generate_docs_figures.py` from synthetic data via `struphy_plots`
itself -- they aren't checked into git, so generate them before building or
running the site locally:

```bash
pip install -e ".[pyvista]" plotly kaleido   # see below for why these two extras
make figures       # renders docs/src/assets/figures/
make docs-dev       # figures + npm run dev
make docs-build     # figures + npm run build
```

- Without the `pyvista` extra (or without a working display), the script
  skips the two PyVista-based figures (`volume.png`, `equilibrium_3d.png`)
  and prints a warning -- but the docs pages that embed them will then fail
  to build, since Astro needs the referenced file to exist. CI installs
  `.[pyvista]` and runs figure generation under `xvfb-run` with Mesa's
  software renderer for exactly this reason (see
  `.github/workflows/docs.yml`); do the same locally if you hit rendering
  errors.
- `plotly`/`kaleido` (static image export) render the `plotly_*.png`
  figures, which demonstrate `array.struphy.data` by plotting the same data
  with Plotly instead of matplotlib. Neither is a `struphy-plots`
  dependency; without them the script skips those figures the same way.

CI regenerates the figures the same way on every push to `devel`, before
deploying to GitHub Pages.

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
