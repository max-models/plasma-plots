"""The Python code blocks of the docs guides are formatted for the docs' narrow code blocks
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
    script = load_script()
    changed, failed = [], []
    for path in sorted([*script.CONTENT.rglob("*.md"), *script.CONTENT.rglob("*.mdx")]):
        _, c, f = script.format_text(path.read_text(), str(path.relative_to(script.ROOT)))
        changed += c
        failed += f
    assert not failed, "docs snippets that don't parse:\n" + "\n".join(failed)
    assert not changed, (
        "run `python scripts/format_docs_snippets.py` to format these docs snippets:\n" + "\n".join(changed)
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
