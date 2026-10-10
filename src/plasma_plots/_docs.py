"""Complete the docstrings of accessor methods with the parameters they inherit.

An accessor method (``array.plasma.plot.lineout``, ``out.plot.energies``, ...) documents only the
parameters that differ from the function it wraps; the first entry of its See Also section names
that function (see CONTRIBUTING.md). The generated API reference fills in the rest; this module
does the same at import time, so ``help()`` on a method shows every parameter too.
"""

from __future__ import annotations

import importlib
import inspect
import re

_UNDERLINE = re.compile(r"^-{3,}$")


def _sections(doc: str) -> list[tuple[str | None, list[str]]]:
    """[(title or None for the text before the first section, lines)] of a cleaned docstring."""
    lines = doc.splitlines()
    sections, title, body, i = [], None, [], 0
    while i < len(lines):
        if (
            i + 1 < len(lines)
            and lines[i].strip()
            and _UNDERLINE.match(lines[i + 1].strip())
            and not lines[i].startswith(" ")
        ):
            sections.append((title, body))
            title, body, i = lines[i].strip(), [], i + 2
            continue
        body.append(lines[i])
        i += 1
    sections.append((title, body))
    return sections


def _join(sections) -> str:
    out = []
    for title, body in sections:
        if title is not None:
            out += [title, "-" * len(title)]
        out += body
    return "\n".join(out).rstrip() + "\n"


def _items(body: list[str]) -> list[tuple[list[str], list[str]]]:
    """[(bare names, lines)] of a Parameters section: an item starts at column 0."""
    items = []
    for line in body:
        if line and not line.startswith(" "):
            head = line.split(" : ", 1)[0]
            items.append(([n.strip().lstrip("*") for n in head.split(",")], [line]))
        elif items:
            items[-1][1].append(line)
    for _, lines in items:  # no blank lines at an item's end
        while lines and not lines[-1].strip():
            lines.pop()
    return items


def _see_also_first(sections) -> str | None:
    for title, body in sections:
        if title and title.lower() == "see also":
            for line in body:
                if line.strip() and not line.startswith(" "):
                    return line.split(" : ", 1)[0].split(",")[0].strip().strip("`")
    return None


def _resolve(name: str, owner):
    """The object ``name`` refers to, seen from a method of the class ``owner``."""
    name = name.lstrip("~").rstrip("()")
    parts = name.split(".")
    module = inspect.getmodule(owner)
    # ClassName.method in the same module, or a function of the same module
    candidate = module
    for part in parts:
        candidate = getattr(candidate, part, None)
        if candidate is None:
            break
    if candidate is not None:
        return candidate
    # a full dotted path: import the longest module prefix
    for split in range(len(parts) - 1, 0, -1):
        try:
            target = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        for part in parts[split:]:
            target = getattr(target, part, None)
            if target is None:
                return None
        return target
    return None


def parameter_docs(function, owner=None, _seen=None) -> dict[str, list[str]]:
    """{bare name: item lines} documented by ``function`` itself or inherited along See Also."""
    doc = inspect.getdoc(function)
    if not doc:
        return {}
    sections = _sections(doc)
    params = {}
    for title, body in sections:
        if title == "Parameters":
            for names, lines in _items(body):
                for name in names:
                    params[name] = lines
    first = _see_also_first(sections)
    _seen = (_seen or set()) | {id(function)}
    if first:
        target = _resolve(first, owner if owner is not None else function)
        target = getattr(target, "__func__", target)
        if callable(target) and id(target) not in _seen:
            owner_of_target = owner if inspect.getmodule(target) is inspect.getmodule(owner) else target
            for name, lines in parameter_docs(target, owner_of_target, _seen).items():
                params.setdefault(name, lines)
    return params


def complete_docstring(function, owner) -> None:
    """Rewrite the Parameters section of ``function`` to cover every parameter of its signature,
    in signature order, with the inherited ones filled in. Leaves it alone if nothing is missing.
    """
    doc = inspect.getdoc(function)
    if not doc:  # e.g. python -OO
        return
    signature = [p for p in inspect.signature(function).parameters.values() if p.name not in ("self", "cls")]
    documented = parameter_docs(function, owner)
    sections = _sections(doc)
    own = next((body for title, body in sections if title == "Parameters"), [])
    own_names = {name for names, _ in _items(own) for name in names}
    if all(p.name in own_names or p.name not in documented for p in signature):
        return
    written, lines = set(), []
    for p in signature:
        item = documented.get(p.name)
        if item is None or id(item) in written:
            continue
        written.add(id(item))
        lines += item
    lines.append("")
    new = [(title, body) for title, body in sections if title != "Parameters"]
    # Parameters goes right after the text, before the other sections
    position = 1 if new and new[0][0] is None else 0
    new.insert(position, ("Parameters", lines))
    if new[0][0] is None and new[0][1] and new[0][1][-1].strip():
        new[0] = (None, new[0][1] + [""])
    function.__doc__ = _join(new)


def complete_class(cls) -> None:
    """Complete every public method of an accessor class (including ``__call__``)."""
    for name, member in vars(cls).items():
        if (not name.startswith("_") or name == "__call__") and inspect.isfunction(member):
            complete_docstring(member, cls)


# ---------------------------------------------------------------------------------------------
# Menus: what an accessor prints, e.g. ``phi.plasma.plot`` in a REPL or notebook
# ---------------------------------------------------------------------------------------------
REFERENCE = "https://max-models.github.io/plasma-plots/reference"
# class name -> (how users reach it, reference page)
MENUS = {
    "ArrayPlots": ("array.plasma.plot", "plot"),
    "ArrayAnalysis": ("array.plasma.analysis", "analysis"),
    "ArrayData": ("array.plasma.data", "data"),
    "SliceView": ("array.plasma.plot.view(...)", "plot"),
    "DatasetPlots": ("dataset.plasma.plot", "dataset"),
    "DatasetAnalysis": ("dataset.plasma.analysis", "dataset"),
    "DatasetData": ("dataset.plasma.data", "dataset"),
    "OutputPlots": ("out.plot", "output"),
    "OutputAnalysis": ("out.analysis", "output"),
    "ProfilePlots": ("out.plot.profile", "output"),
    "PlasmaAccessor": ("array.plasma", ""),
    "PlasmaDatasetAccessor": ("dataset.plasma", ""),
}


def plain(text: str) -> str:
    """reST markup to plain text: ``code`` and :role:`~a.b` become code and b."""
    text = re.sub(
        r":\w+:`~?([^`]+)`",
        lambda m: m.group(1).rsplit(".", 1)[-1] if "~" in m.group(0) else m.group(1),
        text,
    )
    return text.replace("``", "")


def summary(member) -> str:
    """The first line of a docstring, as plain text."""
    doc = inspect.getdoc(member)
    return plain(doc.strip().splitlines()[0]) if doc else ""


def _target(accessor) -> str:
    """A short description of what the accessor works on."""
    for attribute in ("_array", "_dataset"):
        data = getattr(accessor, attribute, None)
        if data is not None:
            sizes = ", ".join(f"{d}: {n}" for d, n in data.sizes.items())
            name = f"{data.name!r} " if getattr(data, "name", None) else ""
            kind = type(data).__name__
            return f"{kind} {name}({sizes})"
    output = getattr(accessor, "_output", None)
    return repr(output) if output is not None else ""


def menu(accessor) -> str:
    """The accessor's methods (and sub-accessors) with their summary lines."""
    cls = type(accessor)
    reached, page = MENUS.get(cls.__name__, (cls.__name__, ""))
    rows = []
    for name, member in vars(cls).items():
        if name.startswith("_") and name != "__call__":
            continue
        if isinstance(member, property) or inspect.isfunction(member):
            function = member.fget if isinstance(member, property) else member
            shown = "()" if name == "__call__" else ("" if isinstance(member, property) else name)
            rows.append((shown if name == "__call__" else name, summary(function)))
    width = max((len(n) for n, _ in rows), default=0)
    target = _target(accessor)
    lines = [f"{reached}" + (f" of {target}" if target else "")]
    lines += [f"  {n.ljust(width)}  {s}" for n, s in rows]
    example = reached.split("(")[0].replace("array", "phi", 1).replace("dataset", "orbits", 1)
    data = getattr(accessor, "_array", getattr(accessor, "_dataset", None))
    name = getattr(data, "name", None)
    if isinstance(name, str) and name.isidentifier() and example.startswith(("phi.", "orbits.")):
        example = name + example[example.index(".") :]  # the array's own name, e.g. b_field.plasma.plot
    first = next((n for n, _ in rows if n != "()"), None)
    if not page:  # array.plasma / dataset.plasma: the sub-accessors
        lines.append(
            f"{example}.plot, .analysis and .data list their methods; python -m plasma_plots prints an overview."
        )
    elif first:
        lines.append(f"help({example}.{first}) shows the parameters; reference: {REFERENCE}/{page}/")
    return "\n".join(lines)


def add_menu(cls) -> None:
    cls.__repr__ = menu
