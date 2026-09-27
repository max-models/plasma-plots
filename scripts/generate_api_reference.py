"""Generate the API reference of the docs site from the docstrings, or check the docstrings.

Reads the package with griffe (statically: nothing is imported), and writes one MDX page per
accessor and per module into ``docs/src/content/docs/reference/``. See "Docstrings and the API
reference" in CONTRIBUTING.md for the docstring rules.

Run from the repo root:

    python scripts/generate_api_reference.py            # write the pages
    python scripts/generate_api_reference.py --check    # list docstrings that break the rules
    python scripts/generate_api_reference.py --check plotting accessors   # only these modules
"""

from __future__ import annotations

import argparse
import ast
import logging
import re
import shutil
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import griffe

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "struphy_plots"
OUT = ROOT / "docs" / "src" / "content" / "docs" / "reference"
BASE = "/struphy-plots/reference"
SOURCE = "https://github.com/struphy-hub/struphy-plots/blob/devel"

# classes users never construct: reached as array.struphy.plot, out.plot, ...
ACCESSOR_CLASSES = {
    "StruphyAccessor", "StruphyDatasetAccessor", "ArrayPlots", "ArrayAnalysis", "ArrayData",
    "DatasetPlots", "DatasetAnalysis", "DatasetData", "SliceView", "OutputPlots", "OutputAnalysis",
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
        m for name, m in obj.members.items()
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
    return [s.value.description for s in sections
            if s.kind.value == "admonition" and (s.title or "").lower() == title.lower()]


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
    ``struphy_plots.plotting.plot_slice`` or ``ArrayPlots.slice`` points to, or None."""
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
        except (KeyError, griffe.AliasResolutionError, griffe.CyclicAliasError, AttributeError, ValueError):
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
        extra = [item.name for item in (own.value if own else []) if bare(item.name) not in names
                 and not all(bare(n.strip()) in names for n in item.name.split(","))]
        if extra:
            problems.append(f"documented parameters not in the signature: {', '.join(extra)}")
        if own:
            untyped = [item.name for item in own.value if not item.annotation and not item.name.startswith("*")]
            if untyped:
                problems.append(f"parameters without a type (name : type): {', '.join(untyped)}")
        if "property" not in obj.labels and returns_value(obj) and not section(sections, "returns") \
                and not section(sections, "yields"):
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
    print(f"{count} problem(s)" if count else "all docstrings follow the rules", file=sys.stderr)
    return 1 if count else 0


# ---------------------------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------------------------
@dataclass
class Page:
    slug: str                 # under reference/, e.g. "plot" or "functions/spectral"
    title: str
    description: str
    classes: list = field(default_factory=list)   # accessor pages: class paths, in order
    module: str | None = None                      # module pages: the module
    intro: str = ""


# How users reach each accessor class; the prefix of its methods on the page.
PREFIXES = {
    "ArrayPlots": "array.struphy.plot", "ArrayAnalysis": "array.struphy.analysis",
    "ArrayData": "array.struphy.data", "SliceView": "view",
    "DatasetPlots": "dataset.struphy.plot", "DatasetAnalysis": "dataset.struphy.analysis",
    "DatasetData": "dataset.struphy.data", "OutputPlots": "out.plot", "ProfilePlots": "out.plot.profile",
    "OutputAnalysis": "out.analysis", "StruphyAccessor": "array.struphy",
    "StruphyDatasetAccessor": "dataset.struphy",
}
# The short heading of a method on a page with several classes.
SHORT = {
    "SliceView": "view", "DatasetPlots": "plot", "DatasetAnalysis": "analysis", "DatasetData": "data",
    "OutputPlots": "plot", "ProfilePlots": "plot.profile", "OutputAnalysis": "analysis",
}
SECTION_TITLES = {
    "SliceView": "SliceView, returned by plot.view()", "DatasetPlots": "dataset.struphy.plot",
    "DatasetAnalysis": "dataset.struphy.analysis", "DatasetData": "dataset.struphy.data",
    "OutputPlots": "out.plot", "ProfilePlots": "out.plot.profile", "OutputAnalysis": "out.analysis",
    "ArrayPlots": "array.struphy.plot",
}

PAGES = [
    Page("plot", "array.struphy.plot", "Plots of one labeled array: time series, slices, animations, 3-D views.",
         classes=["accessors.ArrayPlots", "accessors.SliceView"],
         intro="Plots of one labeled `xarray.DataArray`. Each method selects the dimensions it doesn't "
               "draw by keyword, draws, and returns a `PlotResult` (or an animation or a PyVista plotter)."),
    Page("analysis", "array.struphy.analysis", "Diagnostics of one labeled array: fits, norms, errors, spectra.",
         classes=["accessors.ArrayAnalysis"],
         intro="Numerical diagnostics of one labeled `xarray.DataArray`. They return labeled arrays and "
               "Datasets, so their results can be plotted with `.struphy.plot` in turn."),
    Page("data", "array.struphy.data", "The selected data behind every plot, without drawing it.",
         classes=["accessors.ArrayData"],
         intro="Every plot on `array.struphy.plot` has a twin here that does the same selection and returns "
               "the labeled `xarray` object instead of a figure. See the "
               "[Selecting data](/struphy-plots/guides/data/) guide."),
    Page("dataset", "dataset.struphy", "Plots, diagnostics and data of marker Datasets such as orbits.",
         classes=["accessors.DatasetPlots", "accessors.DatasetAnalysis", "accessors.DatasetData"],
         intro="Plots, diagnostics and data of an `xarray.Dataset` with per-marker variables, such as an "
               "orbits product."),
    Page("output", "out.plot and out.analysis", "Whole-run plots and diagnostics of a Struphy Output.",
         classes=["output_accessors.OutputPlots", "output_accessors.ProfilePlots", "output_accessors.OutputAnalysis"],
         intro="Plots and diagnostics of a whole run, on a Struphy `Output` object. Importing "
               "`struphy_plots` registers them as `out.plot` and `out.analysis`."),
]
MODULE_PAGES = ["plotting", "analysis", "spectral", "spectral_plots", "pyvista_plots", "arrays"]
for name in MODULE_PAGES:
    PAGES.append(Page(f"functions/{name.replace('_', '-')}", f"{PACKAGE}.{name}", "", module=name))
# the analytic theory: an overview (the subpackage docstring) and one page per module
THEORY_PAGES = ["special", "parameters", "kinetic", "waves", "orbits", "exact", "numerics"]
PAGES.append(Page("theory", f"{PACKAGE}.theory", "Analytic theory to compare Struphy runs against.", module="theory"))
for name in THEORY_PAGES:
    PAGES.append(Page(f"theory/{name}", f"{PACKAGE}.theory.{name}", "", module=f"theory.{name}"))


def available_pages(package):
    """The pages whose module exists (a theory module may not be written yet)."""
    out = []
    for page in PAGES:
        if page.module:
            try:
                package[page.module]
            except KeyError:
                continue
        out.append(page)
    return out

# External documentation of types that appear in signatures and parameter types.
EXTERNAL = {
    "xarray.DataArray": "https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html",
    "xr.DataArray": "https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html",
    "xarray.Dataset": "https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html",
    "xr.Dataset": "https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html",
    "matplotlib.axes.Axes": "https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html",
    "matplotlib.figure.Figure": "https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.html",
    "matplotlib.animation.FuncAnimation":
        "https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html",
    "numpy.ndarray": "https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html",
    "pyvista.Plotter": "https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.html",
    "pyvista.StructuredGrid": "https://docs.pyvista.org/api/core/_autosummary/pyvista.StructuredGrid.html",
}


def slugify(text: str) -> str:
    """The heading id Starlight gives a heading (github-slugger)."""
    text = text.lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


class Site:
    """The pages, and where every documented object is anchored, for cross-references."""

    def __init__(self, package):
        self.package = package
        self.anchors = {}      # object path -> url
        self.headings = {}     # object path -> heading text
        self.pages = available_pages(package)
        for page in self.pages:
            used = set()
            for obj, heading in self.entries(page):
                anchor = slugify(heading)
                while anchor in used:
                    anchor += "-1"
                used.add(anchor)
                self.anchors[obj.path] = f"{BASE}/{page.slug}/#{anchor}"
                self.headings[obj.path] = heading
            if page.classes:
                for path in page.classes:
                    cls = package[path]
                    self.anchors.setdefault(cls.path, f"{BASE}/{page.slug}/")
        # exact names a plain ``code`` span may use: full accessor/function paths and class names
        self.names = {}
        for path, url in self.anchors.items():
            obj = package[path.removeprefix(PACKAGE + ".")]
            self.names[self.display_name(obj)] = url
            self.names[obj.path] = url
            if obj.is_class:
                self.names[obj.name] = url

    def entries(self, page):
        """(object, heading text) in page order: classes and their methods, or module members."""
        if page.module:
            module = self.package[page.module]
            for member in own_members(module):
                yield member, member.name
                if member.is_class:
                    for method in own_members(member):
                        yield method, f"{member.name}.{method.name}"
        else:
            many = len(page.classes) > 1
            for path in page.classes:
                cls = self.package[path]
                if many:
                    yield cls, SECTION_TITLES.get(cls.name, cls.name)
                for method in own_members(cls):
                    short = SHORT.get(cls.name) if many and cls.name in SHORT else None
                    if method.name == "__call__":  # out.plot() rather than out.plot.__call__
                        yield method, f"{short or PREFIXES[cls.name].rsplit('.', 1)[-1]}()"
                    else:
                        yield method, f"{short}.{method.name}" if short else method.name

    def url(self, obj):
        return self.anchors.get(obj.path)

    def display_name(self, obj):
        """How users call ``obj``: ``array.struphy.plot.slice`` or ``struphy_plots.spectral.fft``."""
        if obj.parent is not None and obj.parent.is_class and obj.parent.name in PREFIXES:
            name = "__call__" if obj.name == "__call__" else obj.name
            prefix = PREFIXES[obj.parent.name]
            return prefix if name == "__call__" else f"{prefix}.{name}"
        return obj.path


# ---------------------------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------------------------
def escape_mdx(text: str) -> str:
    """Escape what MDX would read as JSX or expressions, outside inline code."""
    parts = re.split(r"(`[^`]*`)", text)
    for i in range(0, len(parts), 2):
        parts[i] = (parts[i].replace("{", "\\{").replace("}", "\\}").replace("<", "&lt;")
                    .replace(">", "&gt;"))
    return "".join(parts)


def markdown(site, text: str, context=None) -> str:
    """reST-flavored docstring text to MDX-safe Markdown, with cross-references as links."""
    if not text:
        return ""
    text = textwrap.dedent(text).strip()

    def role(match):
        target = match.group(1)
        label = None
        if " <" in target and target.endswith(">"):
            label, target = target.split(" <", 1)
            target = target[:-1]
        short = target.startswith("~")
        target = target.lstrip("~")
        obj = resolve(site.package, target, context)
        callable_ = obj is not None and obj.is_function
        if label is None:
            if obj is not None and obj.path in site.headings and obj.parent is not None \
                    and obj.parent.is_class and obj.parent.name in PREFIXES:
                label = site.display_name(obj)
                label = label.rsplit(".", 1)[-1] if short else label
            else:
                label = target.rsplit(".", 1)[-1] if short else target
            label = f"`{label}{'()' if callable_ else ''}`"
        url = site.url(obj) if obj is not None else None
        return f"[{label}]({url})" if url else label

    lines, out, in_code = text.splitlines(), [], False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        line = re.sub(r"``([^`]+)``", r"`\1`", line)
        line = _ROLE.sub(role, line)
        line = re.sub(r"(?<![\[`])`([A-Za-z_][\w.]*)(\(\))?`(?!\]\()", lambda m: autolink(site, m), line)
        # links produced above contain ](url); keep them out of the escaping
        pieces = re.split(r"(\]\([^)]*\))", line)
        out.append("".join(p if p.startswith("](") else escape_mdx(p) for p in pieces))
    return "\n".join(out)


def html_text(text: str) -> str:
    """Text for raw HTML in MDX: escaped, with quotes as entities so the typographer leaves them straight."""
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("{", "&#123;")
            .replace("}", "&#125;").replace('"', "&quot;").replace("'", "&#39;"))


def autolink(site, match):
    """A plain code span that names a documented object exactly, as a link to it."""
    url = site.names.get(match.group(1))
    return f"[{match.group(0)}]({url})" if url else match.group(0)


def type_markdown(site, text: str, context=None) -> str:
    """A type string with known types linked."""
    if not text:
        return ""
    text = str(text)
    tokens = re.split(r"([A-Za-z_][\w.]*)", text)
    out = []
    for token in tokens:
        if not token:
            continue
        if token in EXTERNAL:
            out.append(f'<a href="{EXTERNAL[token]}">{token}</a>')
            continue
        if re.fullmatch(r"[A-Za-z_][\w.]*", token) and token[0].isupper():
            obj = resolve(site.package, token, context)
            if obj is not None and obj.is_class and site.url(obj):
                out.append(f'<a href="{site.url(obj)}">{token}</a>')
                continue
        out.append(html_text(token))
    return "".join(out)


def signature(site, obj, name: str) -> str:
    """The call signature as Python, one parameter per line when long."""
    params, kinds = [], []
    for p in obj.parameters:
        if p.name in ("self", "cls"):
            continue
        kind = p.kind.value
        text = ("**" if kind == "variadic keyword" else "*" if kind == "variadic positional" else "") + p.name
        if p.annotation is not None:
            text += f": {p.annotation}"
        if p.default is not None and kind not in ("variadic keyword", "variadic positional"):
            text += f" = {p.default}" if p.annotation is not None else f"={p.default}"
        params.append(text)
        kinds.append(kind)
    # the bare * before the first keyword-only parameter
    rendered = []
    star_needed = "variadic positional" not in kinds
    for text, kind in zip(params, kinds):
        if kind == "keyword-only" and star_needed:
            rendered.append("*")
            star_needed = False
        rendered.append(text)
    returns = f" -> {obj.returns}" if getattr(obj, "returns", None) is not None else ""
    one_line = f"{name}({', '.join(rendered)}){returns}"
    if len(one_line) <= 90:
        return one_line
    inner = "".join(f"    {p},\n" for p in rendered)
    return f"{name}(\n{inner}){returns}"


def definition_list(items):
    """<dl> of (term html, description markdown)."""
    if not items:
        return ""
    out = ['<dl class="api-list">']
    for term, description in items:
        out.append(f"<dt>{term}</dt>")
        out.append("<dd>\n\n" + (description or "") + "\n\n</dd>")
    out.append("</dl>")
    return "\n".join(out)


def label(text):
    return f'<p class="api-label">{text}</p>'


def raw_items(obj, title: str):
    """The item lines of a docstring section as written: [(name, type)] for ``name : type`` lines,
    or [(None, line)] for bare lines (a Returns type). griffe parses types as expressions, which
    drops quotes and words like ``optional``, so the reference shows these instead."""
    if obj.docstring is None:
        return []
    lines = obj.docstring.value.splitlines()
    items, inside, indent = [], False, None
    for i, line in enumerate(lines):
        stripped = line.strip()
        underline = i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"} and stripped
        if underline:
            inside = stripped.lower() == title.lower()
            indent = None
            continue
        if not inside or not stripped or set(stripped) == {"-"}:
            continue
        current = len(line) - len(line.lstrip())
        if indent is None:
            indent = current
        if current != indent:
            continue
        name, sep, kind = stripped.partition(" : ")
        items.append((name.strip(), kind.strip()) if sep else (None, stripped))
    return items


def raw_type(obj, title: str, name: str, fallback) -> str:
    for item_name, kind in raw_items(obj, title):
        if item_name is not None and name in [bare(n.strip()) for n in item_name.split(",")]:
            return kind
    return str(fallback) if fallback else ""


def render_sections(site, obj, sections, *, parameters=True):
    """The docstring body: description, parameters, returns, raises, see also, notes, examples."""
    out = []
    for s in sections:
        if s.kind.value == "text":
            out.append(markdown(site, s.value, obj))
    if parameters and obj.is_function:
        documented = documented_parameters(site.package, obj, sections)
        items = []
        for name, p in signature_parameters(obj):
            doc = documented.get(bare(name))
            kind = p.kind.value
            term = f'<code class="api-name">{name}</code>'
            if doc is not None and doc.annotation:
                owner = doc_owner(site.package, obj, bare(name))
                shown = raw_type(owner, "Parameters", bare(name), doc.annotation)
                term += f' <span class="api-type">{type_markdown(site, shown, obj)}</span>'
            elif p.annotation is not None:
                term += f' <span class="api-type">{type_markdown(site, p.annotation, obj)}</span>'
            if p.default is not None and kind not in ("variadic keyword", "variadic positional"):
                # a Markdown code span: raw <code> would get curly quotes from the typographer
                term += f' <span class="api-default">= `{p.default}`</span>'
            items.append((term, markdown(site, doc.description, obj) if doc else ""))
        if items:
            out.append(label("Parameters"))
            out.append(definition_list(items))
    elif parameters and obj.is_class:
        s = section(sections, "parameters")
        if s:
            out.append(label("Parameters"))
            out.append(definition_list([
                (f'<code class="api-name">{item.name}</code>'
                 + (f' <span class="api-type">{type_markdown(site, raw_type(obj, "Parameters", item.name, item.annotation), obj)}</span>'
                    if item.annotation else ""),
                 markdown(site, item.description, obj)) for item in s.value]))
    s = section(sections, "attributes")
    if s:
        out.append(label("Attributes"))
        out.append(definition_list([
            (f'<code class="api-name">{item.name}</code>'
             + (f' <span class="api-type">{type_markdown(site, raw_type(obj, "Attributes", item.name, item.annotation), obj)}</span>'
                if item.annotation else ""),
             markdown(site, item.description, obj)) for item in s.value]))
    for kind, title in (("returns", "Returns"), ("yields", "Yields")):
        s = section(sections, kind)
        if s:
            out.append(label(title))
            items = []
            raw = raw_items(obj, title)
            for index, item in enumerate(s.value):
                term = ""
                if item.name:
                    term += f'<code class="api-name">{item.name}</code> '
                shown = item.annotation
                if index < len(raw):
                    shown = raw[index][1] if raw[index][0] is None or raw[index][0] == item.name else shown
                if shown:
                    term += f'<span class="api-type">{type_markdown(site, shown, obj)}</span>'
                items.append((term or "&nbsp;", markdown(site, item.description, obj)))
            out.append(definition_list(items))
    s = section(sections, "raises")
    if s:
        out.append(label("Raises"))
        out.append(definition_list([
            (f'<code class="api-name">{html_text(str(item.annotation))}</code>', markdown(site, item.description, obj))
            for item in s.value]))
    for s in sections:
        if s.kind.value == "admonition" and (s.title or "").lower() not in ("see also",):
            out.append(label(escape_mdx(s.title or "Note")))
            out.append(markdown(site, s.value.description, obj))
    entries = see_also_entries(sections)
    if entries:
        out.append(label("See also"))
        items = []
        for names, description in entries:
            links = []
            for name in names:
                target = resolve(site.package, name, obj)
                shown = site.display_name(target) if target is not None else name
                shown += "()" if target is not None and target.is_function else ""
                url = site.url(target) if target is not None else None
                links.append(f"[`{shown}`]({url})" if url else f"`{shown}`")
            items.append(f"- {', '.join(links)}" + (f": {markdown(site, description, obj)}" if description else ""))
        out.append("\n".join(items))
    s = section(sections, "examples")
    if s:
        out.append(label("Examples"))
        for kind, text in s.value:
            if kind.value == "examples":
                # prompts dropped, and what a line prints shown as a comment below it
                lines = [re.sub(r"^(>>> |\.\.\. |>>>$|\.\.\.$)", "", line) if line.startswith((">>>", "..."))
                         else f"# {line}" for line in text.splitlines() if line.strip()]
                code = "\n".join(lines)
                out.append(f"```python\n{code}\n```")
            else:
                out.append(markdown(site, text, obj))
    return "\n\n".join(part for part in out if part)


def source_link(obj):
    if not obj.lineno:
        return ""
    path = obj.relative_package_filepath if hasattr(obj, "relative_package_filepath") else None
    path = f"src/{PACKAGE}/{path}" if path else f"src/{obj.relative_filepath}"
    path = str(obj.filepath.relative_to(ROOT)) if obj.filepath else path
    lines = f"#L{obj.lineno}" + (f"-L{obj.endlineno}" if obj.endlineno else "")
    return f'<a class="api-source" href="{SOURCE}/{path}{lines}">source</a>'


def render_entry(site, obj, heading: str, level: int) -> str:
    sections, _ = parse(obj)
    hashes = "#" * level
    parts = [f"{hashes} `{heading}`", ""]
    badge = ""
    if obj.is_class:
        badge = "dataclass" if "dataclass" in obj.labels else "class"
    elif "property" in obj.labels:
        badge = "property"
    head = '<div class="api-head">'
    head += f'<span class="api-badge">{badge}</span>' if badge else ""
    head += source_link(obj) + "</div>"
    parts.append(head)
    parts.append("")
    if obj.is_function and "property" not in obj.labels:
        parts.append(f"```python\n{signature(site, obj, site.display_name(obj))}\n```")
    elif obj.is_class and obj.name not in ACCESSOR_CLASSES:
        init = obj.members.get("__init__")
        if init is not None and not init.is_alias:
            parts.append(f"```python\nclass {signature(site, init, obj.path)}\n```")
        else:
            parts.append(f"```python\nclass {obj.path}\n```")
    parts.append("")
    parts.append(render_sections(site, obj, sections))
    return "\n".join(parts)


def frontmatter(title, description, order):
    description = description.replace('"', "'")
    return f'---\ntitle: "{title}"\ndescription: "{description}"\nsidebar:\n  order: {order}\n---\n'


IMPORT_NOTE = """:::note[Where the accessors come from]
The examples on this page use output of a Struphy `Output`, which loads struphy-plots and its
`.struphy` accessors. For other xarray data, run `import struphy_plots` first (see
[Getting started](/struphy-plots/guides/getting-started/#loading-struphy-plots)).
:::
"""
THEORY_NOTE = """:::note[Plain functions]
The theory is plain numpy: no Struphy, no accessors. Import the module, e.g.
`from struphy_plots.theory import kinetic`; its results work directly as `branches=`, `reference=` and
`theory=` of the plots (see the [Theory guide](/struphy-plots/guides/theory/)).
:::
"""
GENERATED = "{/* Generated by scripts/generate_api_reference.py from the docstrings. Do not edit. */}\n"


def summary(site, obj) -> str:
    """The summary line of the docstring (its first line)."""
    if obj.docstring is None:
        return ""
    return markdown(site, obj.docstring.value.strip().splitlines()[0].strip(), obj)


def render_page(site, page, order):
    description = page.description
    intro = page.intro
    if page.module:
        module = site.package[page.module]
        sections, _ = parse(module)
        description = description or (summary(site, module).replace("`", "") or page.title)
        intro = render_sections(site, module, sections, parameters=False)
    note = THEORY_NOTE if (page.module or "").startswith("theory") else IMPORT_NOTE
    parts = [frontmatter(page.title, description, order), GENERATED, note, intro, ""]
    entries = list(site.entries(page))
    # a table of contents, then the entries
    rows = []
    for obj, heading in entries:
        if obj.is_class and page.classes:
            continue
        url = site.url(obj)
        rows.append(f"| [`{heading}`]({url}) | {summary(site, obj)} |")
    if rows:
        kind = "Function" if page.module else "Method"
        parts += ['<div class="api-summary">', "", f"| {kind} | Summary |", "| --- | --- |", *rows, "", "</div>", ""]
    many = page.classes and len(page.classes) > 1
    for obj, heading in entries:
        if page.classes and obj.is_class:
            sections, _ = parse(obj)
            parts.append(f"## {heading}\n")
            parts.append(render_sections(site, obj, sections, parameters=False) + "\n")
            continue
        level = 3 if many or (page.module and obj.parent is not None and obj.parent.is_class) else 2
        parts.append(render_entry(site, obj, heading, level) + "\n")
    return "\n".join(parts)


def render_index(site):
    rows_accessors = []
    for page in site.pages:
        if page.classes:
            names = " and ".join(f"`{name}`" for name in page.title.split(" and "))
            rows_accessors.append(f"| [{names}]({BASE}/{page.slug}/) | {page.description} |")
    rows_modules, rows_theory = [], []
    for page in site.pages:
        if page.module and page.module.startswith("theory"):
            module = site.package[page.module]
            rows_theory.append(f"| [`{page.title}`]({BASE}/{page.slug}/) | {summary(site, module)} |")
        elif page.module:
            module = site.package[page.module]
            rows_modules.append(f"| [`{page.title}`]({BASE}/{page.slug}/) | {summary(site, module)} |")
    return "\n".join([
        frontmatter("API reference", "Every accessor, function and class of struphy-plots.", 0),
        GENERATED,
        IMPORT_NOTE,
        "Generated from the docstrings, so it always matches the code. The guides show the same "
        "functionality with figures; this reference lists every parameter.",
        "",
        "## Accessors",
        "",
        "After `import struphy_plots`, every labeled array and Dataset has a `.struphy` accessor, and "
        "Struphy's `Output` has `.plot` and `.analysis`:",
        "",
        '<div class="api-summary">', "", "| Accessor | Summary |", "| --- | --- |", *rows_accessors, "", "</div>", "",
        "## Functions",
        "",
        "The functions behind the accessors, to call on a plain `xarray.DataArray`:",
        "",
        '<div class="api-summary">', "", "| Module | Summary |", "| --- | --- |", *rows_modules, "", "</div>", "",
        "## Theory",
        "",
        "Analytic results to compare runs against: dispersion relations, plasma parameters, orbits, exact "
        "solutions and the properties of the numerics.",
        "",
        '<div class="api-summary">', "", "| Module | Summary |", "| --- | --- |", *rows_theory, "", "</div>", "",
    ])


def write_pages():
    package = load_package()
    site = Site(package)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "functions").mkdir(parents=True)
    (OUT / "theory").mkdir(parents=True)
    (OUT / "index.mdx").write_text(render_index(site))
    slugs = [page.slug for page in site.pages]
    for order, page in enumerate(site.pages, start=1):
        # a page with pages below it (theory, theory/kinetic) is that directory's index
        nested = any(other.startswith(page.slug + "/") for other in slugs)
        path = OUT / (f"{page.slug}/index.mdx" if nested else f"{page.slug}.mdx")
        path.write_text(render_page(site, page, order))
        print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="check the docstrings instead of writing pages")
    parser.add_argument("modules", nargs="*", help="only these modules (e.g. plotting accessors)")
    args = parser.parse_args()
    if args.check:
        sys.exit(check(set(args.modules) or None))
    write_pages()


if __name__ == "__main__":
    main()
