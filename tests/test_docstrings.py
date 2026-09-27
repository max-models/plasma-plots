"""Every public docstring follows the NumPy-style rules the API reference is generated from
(see "Docstrings and the API reference" in CONTRIBUTING.md)."""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("griffe")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_api_reference.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_api_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_public_docstring_follows_the_rules():
    generator = load_generator()
    package = generator.load_package()
    problems = [
        f"{obj.path}: {problem}"
        for obj in generator.walk(package)
        for problem in generator.check_object(package, obj)
    ]
    assert not problems, "docstrings break the rules (python scripts/generate_api_reference.py --check):\n" + "\n".join(problems)


def test_the_reference_renders_every_page(tmp_path, monkeypatch):
    generator = load_generator()
    monkeypatch.setattr(generator, "OUT", tmp_path / "reference")
    generator.write_pages()
    pages = sorted(p.relative_to(tmp_path / "reference").as_posix() for p in (tmp_path / "reference").rglob("*.mdx"))
    assert "index.mdx" in pages and "plot.mdx" in pages and "functions/spectral.mdx" in pages
    assert "## `slice`" in (tmp_path / "reference" / "plot.mdx").read_text() or \
        "### `slice`" in (tmp_path / "reference" / "plot.mdx").read_text()
