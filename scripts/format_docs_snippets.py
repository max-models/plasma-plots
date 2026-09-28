"""Format the Python code blocks of the docs guides with ``ruff format``.

The docs site shows about 80 characters of code before a block scrolls sideways, so the snippets
are formatted to a shorter line length than the package (``LINE_LENGTH``, not pyproject.toml's
120). Every fenced ```python block in docs/src/content is formatted in place, keeping its
indentation (blocks inside lists or <details>); a trailing comment too long for its line moves
above the statement. A block that doesn't parse is left as it is and
reported.

Run from the repo root:

    python scripts/format_docs_snippets.py            # format in place
    python scripts/format_docs_snippets.py --check    # list the blocks that would change
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import textwrap
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTENT = ROOT / "docs" / "src" / "content"
LINE_LENGTH = 79

# an opening fence (any indentation), the code, and the closing fence at the same indentation
BLOCK = re.compile(
    r"^(?P<indent>[ \t]*)(?P<fence>```+)(?:python|py)\b[^\n]*\n(?P<code>.*?)^(?P=indent)(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)


def hoist_long_comments(code: str) -> str:
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
                if row == start_row and before.strip() and len(before) + 2 + len(token.string) > LINE_LENGTH:
                    moves[row] = col
    except (tokenize.TokenError, SyntaxError):
        return code
    for row, col in sorted(moves.items(), reverse=True):
        line = lines[row - 1]
        indent = line[: len(line) - len(line.lstrip())]
        lines[row - 1 : row] = [indent + line[col:].rstrip() + "\n", line[:col].rstrip() + "\n"]
    return "".join(lines)


def ruff_format(code: str) -> str:
    """``code`` formatted by ``ruff format``; raises ``ValueError`` if it doesn't parse."""
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--isolated", f"--line-length={LINE_LENGTH}", "-"],
        input=code, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip())
    return result.stdout


def format_text(text: str, name: str = "") -> tuple[str, list[str], list[str]]:
    """``text`` with its Python blocks formatted, the blocks that changed and those that failed."""
    changed, failed = [], []

    def replace(match: re.Match) -> str:
        indent, code = match["indent"], match["code"]
        line = text.count("\n", 0, match.start()) + 1
        try:
            formatted = ruff_format(hoist_long_comments(textwrap.dedent(code)))
        except ValueError as error:
            failed.append(f"{name}:{line}: {error.splitlines()[0] if str(error) else 'does not parse'}")
            return match[0]
        formatted = textwrap.indent(formatted, indent, lambda s: s.strip() != "")
        if formatted != code:
            changed.append(f"{name}:{line}")
        return match[0].replace(code, formatted, 1)

    return BLOCK.sub(replace, text), changed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only list the blocks that would change")
    args = parser.parse_args()

    all_changed, all_failed = [], []
    for path in sorted([*CONTENT.rglob("*.md"), *CONTENT.rglob("*.mdx")]):
        text = path.read_text()
        new, changed, failed = format_text(text, str(path.relative_to(ROOT)))
        all_changed += changed
        all_failed += failed
        if changed and not args.check:
            path.write_text(new)

    for entry in all_failed:
        print(f"not formatted (does not parse): {entry}")
    verb = "would reformat" if args.check else "reformatted"
    for entry in all_changed:
        print(f"{verb}: {entry}")
    print(f"{len(all_changed)} block(s) {verb}, {len(all_failed)} skipped")
    return 1 if (args.check and all_changed) or all_failed else 0


if __name__ == "__main__":
    sys.exit(main())
