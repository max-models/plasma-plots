"""Check that every public docstring follows the rules the API reference is built from.

The API reference (starlight-pydocs, see docs/astro.config.mjs) is generated from the NumPy-style
docstrings; see "Docstrings and the API reference" in CONTRIBUTING.md for the rules. This script
lists every public function, method or class whose docstring breaks them. Parameters a wrapper
inherits from the first See Also entry count as documented, as they do in the reference (via
scripts/griffe_extension.py) and in help() (via plasma_plots._docs).

Run from the repo root:

    python scripts/check_docstrings.py                         # the whole package
    python scripts/check_docstrings.py plotting theory.kinetic   # only these modules
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import sys
import textwrap
from pathlib import Path

import griffe

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "plasma_plots"

# classes users never construct: reached as array.plasma.plot, out.plot, ...
ACCESSOR_CLASSES = {
    "PlasmaAccessor",
    "PlasmaDatasetAccessor",
    "ArrayPlots",
    "ArrayAnalysis",
    "ArrayData",
    "DatasetPlots",
    "DatasetAnalysis",
    "DatasetData",
    "SliceView",
    "OutputPlots",
    "OutputAnalysis",
    "ProfilePlots",
}
PUBLIC_DUNDERS = {"__call__"}


# ---------------------------------------------------------------------------------------------
# Loading and walking the package
# ---------------------------------------------------------------------------------------------
def load_package():
    return griffe.load(PACKAGE, search_paths=[str(ROOT / "src")], docstring_parser="numpy")


def is_public(name: str) -> bool:
    return not name.startswith("_") or name in PUBLIC_DUNDERS


def modules(package):
    """Public modules, subpackages' modules included (e.g. ``theory.kinetic``), in name order."""
    found = []
    for name, module in sorted(package.modules.items()):
        if not is_public(name) or module.is_alias:
            continue
        found.append(module)
        if module.is_package:
            found += modules(module)
    return found


def module_name(module) -> str:
    """The name of a module inside the package, e.g. ``plotting`` or ``theory.kinetic``."""
    return module.path.removeprefix(PACKAGE + ".")


def own_members(obj):
    """Public functions and classes defined in a module, or public methods of a class, in source order."""
    members = [
        m
        for name, m in obj.members.items()
        if is_public(name) and not m.is_alias and (m.is_function or m.is_class or "property" in m.labels)
    ]
    return sorted(members, key=lambda m: m.lineno or 0)


def walk(package, only=None):
    """Every public object: modules, their functions and classes, and the classes' methods."""
    for module in modules(package):
        if only and module_name(module) not in only and module.name not in only:
            continue
        yield module
        for member in own_members(module):
            yield member
            if member.is_class:
                yield from own_members(member)


# ---------------------------------------------------------------------------------------------
# Docstring sections
# ---------------------------------------------------------------------------------------------
class _Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def parse(obj):
    """The parsed docstring sections of ``obj`` and griffe's parsing warnings."""
    if obj.docstring is None:
        return [], []
    handler = _Collect()
    logger = logging.getLogger("griffe")
    logger.addHandler(handler)
    try:
        sections = griffe.parse_numpy(obj.docstring, warn_unknown_params=True)
    finally:
        logger.removeHandler(handler)
    return sections, handler.messages


def section(sections, kind):
    for s in sections:
        if s.kind.value == kind:
            return s
    return None


def admonitions(sections, title):
    return [
        s.value.description
        for s in sections
        if s.kind.value == "admonition" and (s.title or "").lower() == title.lower()
    ]


def bare(name: str) -> str:
    return name.lstrip("*")


def signature_parameters(obj):
    """(name with stars, parameter) for every parameter but self/cls."""
    out = []
    for p in obj.parameters:
        if p.name in ("self", "cls"):
            continue
        stars = "**" if p.kind.value == "variadic keyword" else "*" if p.kind.value == "variadic positional" else ""
        out.append((stars + p.name, p))
    return out


_ROLE = re.compile(r":(?:py:)?(?:func|meth|class|attr|mod|obj|data):`([^`]+)`")


def see_also_entries(sections):
    """[(names, description)] from the See Also sections, in order."""
    entries = []
    for text in admonitions(sections, "see also"):
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.startswith((" ", "\t")) and entries:  # a continued description
                names, description = entries[-1]
                entries[-1] = (names, (description + " " + line.strip()).strip())
                continue
            head, _, description = line.partition(" : ")
            if not description and head.rstrip().endswith(" :"):
                head = head.rstrip()[:-2]
            names = []
            for piece in head.split(","):
                piece = piece.strip()
                role = _ROLE.search(piece)
                piece = role.group(1) if role else piece.strip("`")
                if piece:
                    names.append(piece.lstrip("~"))
            entries.append((names, description.strip()))
    return entries


def see_also_targets(sections):
    return [name for names, _ in see_also_entries(sections) for name in names]


def resolve(package, name: str, context=None):
    """The object a reference like ``plot_slice``, ``plotting.plot_slice``,
    ``plasma_plots.plotting.plot_slice`` or ``ArrayPlots.slice`` points to, or None."""
    name = name.strip().lstrip("~").rstrip("()")
    if " <" in name and name.endswith(">"):  # `label <target>`
        name = name.split(" <", 1)[1][:-1]
    for candidate in (name, name.removeprefix(PACKAGE + ".")):
        try:
            obj = package[candidate]
            return obj.final_target if obj.is_alias else obj
        except (KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError):
            pass
    scope = context
    while scope is not None:  # relative to the context: its class, then its module
        try:
            obj = scope[name]
            return obj.final_target if obj.is_alias else obj
        except (
            KeyError,
            griffe.AliasResolutionError,
            griffe.CyclicAliasError,
            AttributeError,
            ValueError,
        ):
            pass
        scope = scope.parent
    for obj in walk(package):  # anywhere, by its last name(s)
        if obj.path.endswith("." + name):
            return obj
    return None


def documented_parameters(package, obj, sections=None, _seen=None):
    """{bare name: DocstringParameter} for ``obj``, including parameters inherited from the first
    See Also entry (the function an accessor method wraps)."""
    sections = parse(obj)[0] if sections is None else sections
    params = {}
    s = section(sections, "parameters")
    if s:
        for item in s.value:
            for name in item.name.split(","):
                params[bare(name.strip())] = item
    targets = see_also_targets(sections)
    _seen = _seen or {obj.path}
    if targets:
        target = resolve(package, targets[0], obj)
        if target is not None and target.path not in _seen and target.is_function:
            inherited = documented_parameters(package, target, _seen=_seen | {target.path})
            for name, item in inherited.items():
                params.setdefault(name, item)
    return params


def doc_owner(package, obj, name, _seen=None):
    """The object whose docstring documents parameter ``name`` of ``obj``: itself, or the function it
    inherits the parameter from."""
    sections = parse(obj)[0]
    own = section(sections, "parameters")
    if own and any(name in [bare(n.strip()) for n in item.name.split(",")] for item in own.value):
        return obj
    targets = see_also_targets(sections)
    _seen = (_seen or set()) | {obj.path}
    if targets:
        target = resolve(package, targets[0], obj)
        if target is not None and target.path not in _seen and target.is_function:
            return doc_owner(package, target, name, _seen)
    return obj


def returns_value(obj) -> bool:
    """Whether the function body has a ``return <value>`` or a ``yield`` of its own."""
    try:
        tree = ast.parse(textwrap.dedent(obj.source))
    except (SyntaxError, OSError, ValueError):
        return False
    function = tree.body[0]

    def visit(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(child, ast.Return) and child.value is not None:
                if not (isinstance(child.value, ast.Constant) and child.value.value is None):
                    return True
            if isinstance(child, (ast.Yield, ast.YieldFrom)):
                return True
            if visit(child):
                return True
        return False

    return visit(function)


# ---------------------------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------------------------
def check_object(package, obj):
    problems = []
    if obj.docstring is None or not obj.docstring.value.strip():
        return ["no docstring"]
    sections, warnings = parse(obj)
    problems += [f"griffe: {w}" for w in warnings if "documented parameter" not in w]
    summary = obj.docstring.value.strip().splitlines()[0].strip()
    if not summary.endswith((".", ":")):
        problems.append(f"summary line should be one sentence ending with a period: {summary[:70]!r}")
    if obj.is_function:
        documented = documented_parameters(package, obj, sections)
        signature = signature_parameters(obj)
        names = {bare(n) for n, _ in signature}
        missing = [n for n, _ in signature if bare(n) not in documented]
        if missing:
            problems.append(f"parameters not documented: {', '.join(missing)}")
        own = section(sections, "parameters")
        extra = [
            item.name
            for item in (own.value if own else [])
            if bare(item.name) not in names and not all(bare(n.strip()) in names for n in item.name.split(","))
        ]
        if extra:
            problems.append(f"documented parameters not in the signature: {', '.join(extra)}")
        if own:
            untyped = [item.name for item in own.value if not item.annotation and not item.name.startswith("*")]
            if untyped:
                problems.append(f"parameters without a type (name : type): {', '.join(untyped)}")
        if (
            "property" not in obj.labels
            and returns_value(obj)
            and not section(sections, "returns")
            and not section(sections, "yields")
        ):
            problems.append("returns a value but has no Returns section")
    elif obj.is_class:
        if "dataclass" in obj.labels:
            fields = [n for n, m in obj.members.items() if m.is_attribute and is_public(n)]
            attributes = section(sections, "attributes")
            listed = {a.name for a in (attributes.value if attributes else [])}
            missing = [f for f in fields if f not in listed]
            if missing:
                problems.append(f"dataclass fields not under Attributes: {', '.join(missing)}")
        elif obj.name not in ACCESSOR_CLASSES and "__init__" in obj.members:
            init = obj.members["__init__"]
            listed = set(documented_parameters(package, obj, sections))
            missing = [n for n, _ in signature_parameters(init) if bare(n) not in listed]
            if missing:
                problems.append(f"constructor parameters not under Parameters: {', '.join(missing)}")
    return problems


def check(only=None) -> int:
    package = load_package()
    count = 0
    for obj in walk(package, only):
        for problem in check_object(package, obj):
            relative = obj.relative_filepath if hasattr(obj, "relative_filepath") else ""
            print(f"{relative}:{obj.lineno or 1}: {obj.path.removeprefix(PACKAGE + '.')}: {problem}")
            count += 1
    print(
        f"{count} problem(s)" if count else "all docstrings follow the rules",
        file=sys.stderr,
    )
    return 1 if count else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "modules",
        nargs="*",
        help="only these modules (e.g. plotting accessors theory.kinetic)",
    )
    args = parser.parse_args()
    sys.exit(check(set(args.modules) or None))


if __name__ == "__main__":
    main()
