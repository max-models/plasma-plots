"""A griffe extension for the API reference: inherited parameter docs, and Sphinx roles as links.

An accessor method (``array.plasma.plot.lineout``, ``out.plot.energies``, ...) documents only the
parameters that differ from the function it wraps; the first entry of its See Also section names
that function (see CONTRIBUTING.md). This extension fills in the rest while griffe loads the
package, so the API reference (starlight-pydocs, or anything else built on griffe) lists every
parameter. ``plasma_plots._docs`` does the same at import time for ``help()``.

The docstrings cross-reference with Sphinx roles (``:func:`plot_slice```, ``:meth:`ArrayPlots.view```,
``:class:`~plasma_plots.plotting.PlotResult```), which read well in ``help()``. The reference renders
mkdocstrings-style references instead, so the extension also rewrites each role, and each name
under See Also, to ``[`label`][full.dotted.path]``, resolving short and relative names.

It is self-contained (griffe and the standard library only), because griffe may run in its own
environment, e.g. ``uvx --from griffe griffe dump -e scripts/griffe_extension.py``.
"""

from __future__ import annotations

import re

import griffe

_UNDERLINE = re.compile(r"^-{3,}$")
_ROLE = re.compile(r":(?:py:)?(?:func|meth|class|attr|obj):`~?([^`]+)`")


def _sections(doc: str):
    """[(title or None for the text before the first section, lines)] of a NumPy docstring."""
    lines = doc.splitlines()
    sections, title, body, i = [], None, [], 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i].strip() and not lines[i].startswith(" ") \
                and _UNDERLINE.match(lines[i + 1].strip()):
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


def _items(body):
    """[(bare names, lines)] of a Parameters section, without trailing blank lines."""
    items = []
    for line in body:
        if line and not line.startswith(" "):
            head = line.split(" : ", 1)[0]
            items.append(([name.strip().lstrip("*") for name in head.split(",")], [line]))
        elif items:
            items[-1][1].append(line)
    for _, lines in items:
        while lines and not lines[-1].strip():
            lines.pop()
    return items


def _first_see_also(sections):
    for title, body in sections:
        if title and title.lower() == "see also":
            for line in body:
                if line.strip() and not line.startswith(" "):
                    head = line.split(" : ", 1)[0].split(",")[0].strip()
                    role = _ROLE.search(head)
                    return (role.group(1) if role else head.strip("`")).lstrip("~")
    return None


def _resolve(pkg, name: str, context):
    """The function a See Also name refers to: a full or package-relative path, a name relative to
    the context's class or module, or a unique suffix anywhere in the package."""
    name = name.rstrip("()")
    for candidate in (name, name.removeprefix(pkg.name + ".")):
        try:
            obj = pkg[candidate]
            return obj.final_target if obj.is_alias else obj
        except (KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError):
            pass
    scope = context.parent
    while scope is not None:
        try:
            obj = scope[name]
            return obj.final_target if obj.is_alias else obj
        except (KeyError, ValueError, griffe.AliasResolutionError, griffe.CyclicAliasError):
            pass
        scope = scope.parent
    return None


def _documented(pkg, obj, seen):
    """{bare name: item lines} documented by ``obj`` or inherited along the first See Also entries."""
    if obj.docstring is None or obj.path in seen:
        return {}
    sections = _sections(obj.docstring.value)
    params = {}
    for title, body in sections:
        if title == "Parameters":
            for names, lines in _items(body):
                for name in names:
                    params[name] = lines
    first = _first_see_also(sections)
    if first:
        target = _resolve(pkg, first, obj)
        if target is not None and target.is_function:
            for name, lines in _documented(pkg, target, seen | {obj.path}).items():
                params.setdefault(name, lines)
    return params


_ANY_ROLE = re.compile(r":(?:py:)?(func|meth|class|attr|mod|obj|data|exc):`(~?)([^`<]+?)(?: <([^`>]+)>)?`")


def _link(pkg, context, name, short=False, label=None, role="obj"):
    """``[`label`][target]`` for a name, resolved in the package if possible."""
    name = name.strip().rstrip("()")
    target = _resolve(pkg, name, context)
    path = target.path if target is not None else name
    if label is None:
        label = name.rsplit(".", 1)[-1] if short else name
        if (target is not None and target.is_function) or (target is None and role in ("func", "meth")):
            label += "()"
    return f"[`{label}`][{path}]"


def _rewrite(pkg, obj, text: str) -> str:
    """Roles and See Also names as mkdocstrings references, outside code blocks."""
    out, in_code, in_see_also = [], False, False
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
        if in_code or stripped.startswith(">>>") or stripped.startswith("..."):
            out.append(line)
            continue
        if i + 1 < len(lines) and _UNDERLINE.match(lines[i + 1].strip() or "x") and not line.startswith(" "):
            in_see_also = stripped.lower() == "see also"
        elif in_see_also and line and not line.startswith(" ") and not _UNDERLINE.match(stripped):
            head, sep, rest = line.partition(" : ")
            names = [n.strip() for n in head.split(",")]
            if all(n and not n.startswith(("[", ":")) for n in names):
                line = ", ".join(_link(pkg, obj, n.strip("`")) for n in names) + (sep + rest if sep else "")
        line = _ANY_ROLE.sub(lambda m: _link(pkg, obj, m.group(4) or m.group(3), short=bool(m.group(2)),
                                             label=m.group(3) if m.group(4) else None, role=m.group(1)), line)
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def _documented_objects(obj):
    """Every object with a docstring: modules, classes, functions, attributes."""
    for member in obj.members.values():
        if member.is_alias:
            continue
        yield member
        if member.is_class or member.is_module:
            yield from _documented_objects(member)


def _functions(obj):
    for member in obj.members.values():
        if member.is_alias:
            continue
        if member.is_function:
            yield member
        elif member.is_class or member.is_module:
            yield from _functions(member)


class InheritParameters(griffe.Extension):
    """Complete the Parameters section of every function that inherits parameter docs."""

    def on_package(self, *, pkg, loader, **kwargs):  # noqa: ARG002  (griffe's signature)
        completed = {}
        for function in _functions(pkg):
            new = self._completed(pkg, function)
            if new is not None:
                completed[function] = new
        # replace the docstrings only now, so every lookup above read the originals
        texts = {}
        for obj in [pkg, *_documented_objects(pkg)]:
            if obj.docstring is not None:
                texts[obj] = _rewrite(pkg, obj, completed.get(obj, obj.docstring.value))
        for obj, text in texts.items():
            old = obj.docstring
            obj.docstring = griffe.Docstring(
                text, lineno=old.lineno, endlineno=old.endlineno, parent=obj,
                parser=old.parser, parser_options=old.parser_options,
            )

    @staticmethod
    def _completed(pkg, function):
        if function.docstring is None:
            return None
        signature = [p.name for p in function.parameters if p.name not in ("self", "cls")]
        sections = _sections(function.docstring.value)
        own = {n for title, body in sections if title == "Parameters" for names, _ in _items(body) for n in names}
        documented = _documented(pkg, function, set())
        if all(name in own or name not in documented for name in signature):
            return None
        lines, written = [], set()
        for name in signature:
            item = documented.get(name)
            if item is not None and id(item) not in written:
                written.add(id(item))
                lines += item
        lines.append("")
        rest = [(title, body) for title, body in sections if title != "Parameters"]
        position = 1 if rest and rest[0][0] is None else 0
        if position and rest[0][1] and rest[0][1][-1].strip():
            rest[0] = (None, rest[0][1] + [""])
        rest.insert(position, ("Parameters", lines))
        return _join(rest)
