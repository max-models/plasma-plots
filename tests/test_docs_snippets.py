"""The Python code the docs site shows is formatted for its narrow code blocks: the guides' code
blocks, the scripts they include and the docstring examples of the API reference
(``python scripts/format_docs_snippets.py`` formats them)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "format_docs_snippets.py"

if subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True).returncode != 0:
    pytest.skip("ruff is not installed", allow_module_level=True)


def load_script():
    spec = importlib.util.spec_from_file_location("format_docs_snippets", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_docs_snippet_is_formatted():
    # the guides' code blocks, the scripts they include and the docstring examples
    changed, failed = load_script().format_all(write=False)
    assert not failed, "docs snippets that don't parse:\n" + "\n".join(failed)
    assert not changed, "run `python scripts/format_docs_snippets.py` to format these docs snippets:\n" + "\n".join(
        changed
    )


def test_a_long_trailing_comment_moves_above_its_statement():
    script = load_script()
    code = "x = f(alpha, beta, gamma)  # a comment that no longer fits on the same line as its code\n"
    assert script.format_text(f"```python\n{code}```\n")[0] == (
        "```python\n# a comment that no longer fits on the same line as its code\nx = f(alpha, beta, gamma)\n```\n"
    )


def test_an_indented_block_keeps_its_indentation():
    script = load_script()
    text = "1. Step:\n\n   ```python\n   f(x = 1)\n   ```\n"
    assert script.format_text(text)[0] == "1. Step:\n\n   ```python\n   f(x=1)\n   ```\n"


def test_a_docstring_example_fits_with_its_prompt():
    script = load_script()
    call = "f(" + ", ".join(f"parameter_{i}={i}.0" for i in range(6)) + ")"  # 97 characters
    lines = script.format_examples(f"    >>> {call}\n    3\n")[0].splitlines()
    assert lines[0] == "    >>> f(" and lines[-1] == "    3"  # the expected output stays
    assert all(line.startswith("    ... ") or line == "    ... )" for line in lines[1:-1])
    assert max(len(line) - 4 for line in lines) <= script.LINE_LENGTH  # as rendered, unindented


def test_each_statement_of_an_example_gets_its_own_prompt():
    script = load_script()
    code = "x = f(alpha, beta, gamma)  # a comment that is too long to stay where it is now"
    assert script.format_examples(f">>> {code}\n")[0] == (
        ">>> # a comment that is too long to stay where it is now\n>>> x = f(alpha, beta, gamma)\n"
    )


def test_the_included_scripts_are_found():
    names = {path.name for path in load_script().included_scripts()}
    assert {
        "generate_gvec_figures.py",
        "generate_desc_figures.py",
        "generate_real_example_figures.py",
    } <= names
