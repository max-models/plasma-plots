"""Write the API index (``python -m struphy_plots api``) to ``API.md`` and the docs site's ``/llms-api.txt``.

Run it after adding, removing or changing an accessor method or public function;
``tests/test_api_index.py`` fails until ``API.md`` matches the code.

    python scripts/api_index.py
"""

from pathlib import Path

from struphy_plots._api import api_index

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [ROOT / "API.md", ROOT / "docs" / "public" / "llms-api.txt"]


def main():
    text = api_index()
    for path in TARGETS:
        path.write_text(text)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
