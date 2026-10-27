"""Every version in pyproject.toml has its release notes: publishing on main needs releases/<version>.md."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_the_current_version_has_release_notes():
    # tomllib is 3.11+: read the version by pattern
    version = re.search(
        r'^version = "(.+)"$', (ROOT / "pyproject.toml").read_text(), re.M
    ).group(1)
    notes = ROOT / "releases" / f"{version}.md"
    # .github/workflows/publish.yml stops the release without this file
    assert notes.is_file(), f"missing {notes.relative_to(ROOT)} for version {version}"
    assert notes.read_text().splitlines()[0] == f"# plasma-plots {version}"
