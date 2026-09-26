.PHONY: install test lint figures docs-install docs-build docs-dev clean

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests scripts

# Runs real struphy simulations and renders docs/src/assets/figures/*.png from
# their output (needs the full compiled struphy runtime, not just
# struphy-plots -- see CONTRIBUTING.md). Not checked into git -- run this
# before building the docs site. Takes a couple of minutes.
figures:
	python3 scripts/generate_docs_figures.py

docs-install:
	cd docs && npm ci

docs-build: figures
	cd docs && npm run build

docs-dev: figures
	cd docs && npm run dev

clean:
	rm -rf docs/dist docs/.astro docs/src/assets/figures docs/public/figures docs/public/plotly
	find . -name '__pycache__' -not -path './docs/node_modules/*' -exec rm -rf {} +
