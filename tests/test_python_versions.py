"""Python 3.10 is supported: the code parses as 3.10, and the metadata, CI and docs say so."""

import ast
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MINIMUM = (3, 10)
SOURCES = sorted(
    p for folder in ("src", "tests", "scripts") for p in (ROOT / folder).rglob("*.py") if "__pycache__" not in p.parts
)


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_source_parses_with_the_oldest_supported_grammar(path):
    # feature_version rejects newer syntax (except*, PEP 695 generics, ...) even when the tests
    # run on a newer Python, so every CI job checks it, not only the 3.10 one
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=MINIMUM)


def test_pyproject_declares_the_minimum_version():
    # tomllib is 3.11+: read the two fields by pattern
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^requires-python = ">=3\.10"$', text, re.M)
    assert '"Programming Language :: Python :: 3.10"' in text


def test_ci_and_docs_include_the_minimum_version():
    workflow = (ROOT / ".github/workflows/test.yml").read_text()
    matrix = re.search(r"python-version: \[(.*)\]", workflow).group(1)
    assert '"3.10"' in matrix
    for doc in ("README.md", "docs/src/content/docs/guides/getting-started.mdx"):
        assert "Python **3.10 or newer** is required" in (ROOT / doc).read_text()


def test_running_python_is_supported():
    assert sys.version_info[:2] >= MINIMUM
