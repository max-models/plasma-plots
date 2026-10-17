.PHONY: install test lint figures gallery docs-install docs-build docs-dev clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

# Renders docs/src/assets/figures/*.png (synthetic data, fast),
# docs/src/assets/figures/real_*.png (a real struphy simulation, needs the
# full compiled struphy runtime), gvec_*.png (real GVEC equilibria, needs
# pip install gvec) and desc_*.png (DESC's example equilibria, needs
# pip install desc-opt) -- see CONTRIBUTING.md. Not checked into git -- run this
# before building the docs site.
figures:
	python3 scripts/api_index.py
	python3 scripts/generate_docs_figures.py
	python3 scripts/generate_real_example_figures.py
	python3 scripts/generate_gvec_figures.py
	python3 scripts/generate_desc_figures.py

# Runs the scripts in examples/ that changed (FORCE=1 reruns all) for the Gallery pages;
# `starlight-gallery convert`, which writes the pages, runs as part of the docs build.
gallery:
	cd docs && DOCS_PYTHON=$(shell command -v python3) npx starlight-gallery execute $(if $(FORCE),--force,)

docs-install:
	cd docs && npm ci

docs-build: figures gallery
	cd docs && npm run build

# the link check runs in docs-build only (and in CI), not on every reload
docs-dev: figures gallery
	cd docs && DOCS_VALIDATE_LINKS=false npm run dev

clean:
	rm -rf docs/dist docs/.astro docs/.cache docs/public/examples docs/src/content/docs/examples docs/src/assets/figures docs/public/figures docs/public/plotly docs/public/pyvista docs/public/llms-api.txt
	find . -name '__pycache__' -not -path './docs/node_modules/*' -exec rm -rf {} +
