.PHONY: install test lint figures docs-install docs-build docs-dev clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

# Renders docs/src/assets/figures/*.png (synthetic data, fast) and
# docs/src/assets/figures/real_*.png (a real struphy simulation, needs the
# full compiled struphy runtime -- see CONTRIBUTING.md). Not checked into
# git -- run this before building the docs site.
figures:
	python3 scripts/generate_docs_figures.py
	python3 scripts/generate_real_example_figures.py

docs-install:
	cd docs && npm ci

docs-build: figures
	cd docs && npm run build

docs-dev: figures
	cd docs && npm run dev

clean:
	rm -rf docs/dist docs/.astro docs/src/assets/figures docs/public/figures docs/public/plotly
	find . -name '__pycache__' -not -path './docs/node_modules/*' -exec rm -rf {} +
