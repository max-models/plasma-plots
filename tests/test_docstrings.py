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


def test_help_shows_every_parameter_of_every_accessor_method():
    """The docstrings are completed at import time with the parameters a method inherits, the
    same ones the generated reference shows."""
    import inspect

    from struphy_plots import accessors, output_accessors
    from struphy_plots._docs import _items, _sections

    generator = load_generator()
    package = generator.load_package()
    classes = [getattr(accessors, n) for n in ("ArrayPlots", "ArrayAnalysis", "ArrayData", "SliceView",
                                              "DatasetPlots", "DatasetAnalysis", "DatasetData")]
    classes += [getattr(output_accessors, n) for n in ("OutputPlots", "OutputAnalysis", "ProfilePlots")]
    problems = []
    for cls in classes:
        for name, method in vars(cls).items():
            if (name.startswith("_") and name != "__call__") or not inspect.isfunction(method):
                continue
            signature = {p.name for p in inspect.signature(method).parameters.values() if p.name != "self"}
            sections = _sections(inspect.getdoc(method))
            shown = {n for title, body in sections if title == "Parameters" for names, _ in _items(body) for n in names}
            if signature - shown:
                problems.append(f"{cls.__name__}.{name}: help() misses {sorted(signature - shown)}")
            reference = set(generator.documented_parameters(package, package[f"{cls.__module__.split('.', 1)[1]}.{cls.__name__}.{name}"]))
            if (signature & shown) != (signature & reference):
                problems.append(f"{cls.__name__}.{name}: help() and the reference differ")
    assert not problems, "\n".join(problems)
