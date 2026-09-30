"""Format the Python code the docs site shows with ``ruff format``, for its narrow code blocks.

The docs site shows about 80 characters of code before a block scrolls sideways, so this code is
formatted to a shorter line length than the package (``LINE_LENGTH``, not pyproject.toml's 120).
Three kinds of code are formatted in place:

* every fenced ```python block of the guides in docs/src/content, keeping its indentation
  (blocks inside lists or <details>);
* the scripts the guides include whole or in parts (``import script from '...py?raw'``), such
  as scripts/generate_gvec_figures.py;
* the examples (``>>>`` and ``...`` lines) of the docstrings in src/, which the API reference
  shows with their prompts: each example is formatted to ``LINE_LENGTH`` minus the prompt, each
  statement gets its own ``>>>``, and the rest of the package's code is left alone.

A trailing comment too long for its line moves above the statement. Code that doesn't parse is
left as it is and reported.

Run from the repo root:

    python scripts/format_docs_snippets.py            # format in place
    python scripts/format_docs_snippets.py --check    # list what would change
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import textwrap
import tokenize
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "docs" / "src" / "content"
PACKAGE = ROOT / "src" / "plasma_plots"
LINE_LENGTH = 79
PROMPT = 4  # ">>> " and "... "

# an opening fence (any indentation), the code, and the closing fence at the same indentation
BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>```+)(?:python|py)\b[^\n]*\n(?P<code>.*?)^(?P=indent)(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
# a script a guide includes: import script from '../../../../../scripts/generate_gvec_figures.py?raw'
INCLUDE = re.compile(
    r"""^import\s+\w+\s+from\s+['"](?P<path>[^'"]+\.py)\?raw['"]""", re.MULTILINE
)
# a docstring example: a ">>> " line and the "... " lines that continue it, at the same indentation
EXAMPLE = re.compile(
    r"^(?P<indent>[ \t]*)>>> .*\n(?:(?P=indent)\.\.\.(?: .*)?\n)*", re.MULTILINE
)


def hoist_long_comments(code: str, line_length: int = LINE_LENGTH) -> str:
    """``code`` with each trailing comment that doesn't fit the line moved above its statement.

    Left in place, ruff would split the call to make room: ``f(\\n    x\\n)  # comment``.
    Only comments on one-line statements are moved; a comment inside brackets stays.
    """
    lines = code.splitlines(keepends=True)
    moves = {}  # row -> column of the comment
    depth, start_row = 0, 1
    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.OP and token.string in "([{":
                depth += 1
            elif token.type == tokenize.OP and token.string in ")]}":
                depth -= 1
            elif token.type in (tokenize.NEWLINE, tokenize.NL) and depth == 0:
                start_row = token.end[0] + 1
            elif token.type == tokenize.COMMENT and depth == 0:
                row, col = token.start
                before = lines[row - 1][:col].rstrip()
                if (
                    row == start_row
                    and before.strip()
                    and len(before) + 2 + len(token.string) > line_length
                ):
                    moves[row] = col
    except (tokenize.TokenError, SyntaxError):
        return code
    for row, col in sorted(moves.items(), reverse=True):
        line = lines[row - 1]
        indent = line[: len(line) - len(line.lstrip())]
        lines[row - 1 : row] = [
            indent + line[col:].rstrip() + "\n",
            line[:col].rstrip() + "\n",
        ]
    return "".join(lines)


def ruff_format(code: str, line_length: int = LINE_LENGTH) -> str:
    """``code`` formatted by ``ruff format``; raises ``ValueError`` if it doesn't parse."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--isolated",
            f"--line-length={line_length}",
            "-",
        ],
        input=code,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip())
    return result.stdout


def format_code(code: str, line_length: int = LINE_LENGTH) -> str:
    """``code`` with long trailing comments moved up, then formatted by ``ruff format``."""
    return ruff_format(hoist_long_comments(code, line_length), line_length)


def _failure(name: str, line: int, error: ValueError) -> str:
    return f"{name}:{line}: {str(error).splitlines()[0] if str(error) else 'does not parse'}"


def format_text(text: str, name: str = "") -> tuple[str, list[str], list[str]]:
    """``text`` with its Python blocks formatted, the blocks that changed and those that failed."""
    changed, failed = [], []

    def replace(match: re.Match) -> str:
        indent, code = match["indent"], match["code"]
        line = text.count("\n", 0, match.start()) + 1
        try:
            formatted = format_code(textwrap.dedent(code))
        except ValueError as error:
            failed.append(_failure(name, line, error))
            return match[0]
        formatted = textwrap.indent(formatted, indent, lambda s: s.strip() != "")
        if formatted != code:
            changed.append(f"{name}:{line}")
        return match[0].replace(code, formatted, 1)

    return BLOCK.sub(replace, text), changed, failed


def format_examples(text: str, name: str = "") -> tuple[str, list[str], list[str]]:
    """``text`` (a module's source) with its docstring examples formatted, as :func:`format_text`.

    Each ``>>>`` statement with its ``...`` lines is formatted on its own, so that with its prompt
    it fits ``LINE_LENGTH``; expected output lines and all other code stay as they are.
    """
    changed, failed = [], []

    def replace(match: re.Match) -> str:
        indent, source = match["indent"], match[0]
        lines = source.splitlines()
        code = "\n".join(line[len(indent) + PROMPT :] for line in lines) + "\n"
        line = text.count("\n", 0, match.start()) + 1
        try:
            formatted = format_code(code, LINE_LENGTH - PROMPT)
        except ValueError as error:
            failed.append(_failure(name, line, error))
            return source
        rows = formatted.rstrip("\n").split("\n")
        new = "".join(
            f"{indent}{_prompt(rows, i)}{' ' + row if row else ''}\n"
            for i, row in enumerate(rows)
        )
        if new != source:
            changed.append(f"{name}:{line}")
        return new

    return EXAMPLE.sub(replace, text), changed, failed


CONTINUATIONS = (")", "]", "}", "else", "elif", "except", "finally", "case")


def _prompt(rows: list[str], i: int) -> str:
    """``>>>`` for a row that starts a statement (or is a comment of its own), else ``...``."""
    row = rows[i]
    if i == 0:
        return ">>>"
    starts = row and not row[0].isspace() and not row.startswith(CONTINUATIONS)
    return ">>>" if starts and _complete("\n".join(rows[:i])) else "..."


def _complete(code: str) -> bool:
    """Whether ``code`` ends outside brackets and strings, where a new statement can start."""
    try:
        list(tokenize.generate_tokens(io.StringIO(code + "\n").readline))
    except (tokenize.TokenError, SyntaxError):
        return False
    return True


def included_scripts() -> list[Path]:
    """The scripts the guides include with ``?raw`` imports."""
    scripts = set()
    for page in [*CONTENT.rglob("*.md"), *CONTENT.rglob("*.mdx")]:
        for match in INCLUDE.finditer(page.read_text()):
            scripts.add((page.parent / match["path"]).resolve())
    return sorted(scripts)


def format_script(text: str, name: str = "") -> tuple[str, list[str], list[str]]:
    """A whole script formatted, as :func:`format_text`."""
    try:
        formatted = format_code(text)
    except ValueError as error:
        return text, [], [_failure(name, 1, error)]
    return formatted, [name] if formatted != text else [], []


def targets():
    """``(path, formatter)`` for everything this script formats."""
    for path in sorted([*CONTENT.rglob("*.md"), *CONTENT.rglob("*.mdx")]):
        yield path, format_text
    for path in included_scripts():
        yield path, format_script
    for path in sorted(PACKAGE.rglob("*.py")):
        yield path, format_examples


def format_all(*, write: bool) -> tuple[list[str], list[str]]:
    """Format every target (in threads: each waits on its ruff processes); what changed and failed."""

    def one(target):
        path, formatter = target
        text = path.read_text()
        new, changed, failed = formatter(text, str(path.relative_to(ROOT)))
        if changed and write:
            path.write_text(new)
        return changed, failed

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, targets()))
    return [c for changed, _ in results for c in changed], [
        f for _, failed in results for f in failed
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check", action="store_true", help="only list what would change"
    )
    args = parser.parse_args()

    all_changed, all_failed = format_all(write=not args.check)

    for entry in all_failed:
        print(f"not formatted (does not parse): {entry}")
    verb = "would reformat" if args.check else "reformatted"
    for entry in all_changed:
        print(f"{verb}: {entry}")
    print(f"{len(all_changed)} block(s) {verb}, {len(all_failed)} skipped")
    return 1 if (args.check and all_changed) or all_failed else 0


if __name__ == "__main__":
    sys.exit(main())
