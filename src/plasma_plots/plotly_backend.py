"""Interactive Plotly versions of the plots: ``backend="plotly"``.

Every accessor plot that draws with Matplotlib (``array.plasma.plot.*``,
``dataset.plasma.plot.*``, ``out.plot.*``) takes ``backend="plotly"``. The plot is drawn with
Matplotlib as usual, off screen, and the drawn figure is converted into a Plotly figure: the same
data, color limits, fits, reference curves, labels and layout, but with hover values, zoom and,
for animations and viewers, a slider in the browser. The two backends cannot disagree about what
they show, because there is only one drawing code path.

>>> result = phi.plasma.plot.slice(
...     coords="physical", plane="XY", t=-1, eta3=0, backend="plotly"
... )
>>> result.fig  # a plotly.graph_objects.Figure
>>> result.save("phi.html")  # a standalone page; .png/.pdf/.svg need kaleido
>>> # the default for every plot from now on
>>> plasma_plots.set_backend("plotly")

Plots return a :class:`~plasma_plots.plotting.PlotResult` with either backend (``fit_results``
and ``data`` included); animations and viewers return one too, whose figure has a slider. Plotly
is optional: ``pip install "plasma-plots[plotly]"``.

The conversion itself is also available for any Matplotlib figure drawn with the plotting
functions: :func:`to_plotly`, :func:`animation_to_plotly` and
:meth:`~plasma_plots.plotting.PlotResult.to_plotly`.

What converts: lines, markers and scatters (also colored by a value), meshes (heatmaps on
rectilinear grids; on mapped, curvilinear grids an image of the mesh with the values under the
cursor), contour lines, quivers, colored line collections, horizontal and vertical lines and
bands, annotations, colorbars, legends, log axes, twin axes, 3-D lines and scatters, shared and
equal-aspect axes, and titles. Mathtext labels (``$\\omega$``, ``$p_0$``) become Unicode and
sub/superscripts. Anything else is left out with a :class:`ConversionWarning`.
"""

from __future__ import annotations

import base64
import contextlib
import contextvars
import functools
import inspect
import io
import re
import warnings

import numpy as np

BACKENDS = ("matplotlib", "plotly", "tikz")
PX_PER_INCH = 100.0
PX_PER_PT = PX_PER_INCH / 72.0
#: The longer side, in pixels, of the image a mesh on a mapped (curvilinear) grid is drawn as. Lower it
#: for smaller pages, e.g. of long animations: ``plasma_plots.plotly_backend.IMAGE_PIXELS = 600``.
IMAGE_PIXELS = 1200

_default = "matplotlib"
# inside a plot method: nested accessor calls draw with Matplotlib (the outermost call converts)
_drawing = contextvars.ContextVar("plasma_plots_drawing", default=False)
# how many plot methods are running, one inside the other
_depth = contextvars.ContextVar("plasma_plots_depth", default=0)
# the result lists of the figures being composed with plasma_plots.figure, innermost last
_collecting: list[list] = []


class ConversionWarning(UserWarning):
    """A part of a Matplotlib figure that has no Plotly counterpart here and is left out."""


# ---------------------------------------------------------------------------------------------
# Choosing the backend
# ---------------------------------------------------------------------------------------------
def set_backend(backend: str) -> str:
    """Set the backend of every plot that does not pass ``backend=`` itself.

    Parameters
    ----------
    backend : {"matplotlib", "plotly", "tikz"}
        The new default (``"tikz"``: see :mod:`plasma_plots.tikz_backend`).

    Returns
    -------
    str
        The previous default, e.g. to restore it afterwards.

    Raises
    ------
    ValueError
        If ``backend`` is not ``"matplotlib"``, ``"plotly"`` or ``"tikz"``.

    See Also
    --------
    get_backend : The current default.

    Examples
    --------
    >>> previous = plasma_plots.set_backend("plotly")
    >>> phi.plasma.plot.slice(t=-1, eta3=0)  # a Plotly figure
    >>> plasma_plots.set_backend(previous)
    """
    global _default
    previous, _default = _default, _check(backend)
    return previous


def get_backend() -> str:
    """Return the backend of plots that do not pass ``backend=`` themselves.

    Returns
    -------
    str
        ``"matplotlib"`` (the default), ``"plotly"`` or ``"tikz"``.

    See Also
    --------
    set_backend : Changes it.
    """
    return _default


def _check(backend):
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    return backend


def resolve_backend(backend: str | None) -> str:
    """The backend a plot draws with: ``backend``, or the default if it is ``None``.

    Inside another plot (e.g. ``ArrayPlots.slice`` calling ``SliceView.slice``) it is always
    ``"matplotlib"``: the outermost call converts the finished figure.

    Parameters
    ----------
    backend : {"matplotlib", "plotly", "tikz"} or None
        The backend asked for.

    Returns
    -------
    str
        ``"matplotlib"``, ``"plotly"`` or ``"tikz"``.
    """
    if _drawing.get():
        return "matplotlib"
    return _default if backend is None else _check(backend)


def _plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            'backend="plotly" needs plotly: pip install "plasma-plots[plotly]" (or pip install plotly)'
        ) from error
    return go


def with_backend(method):
    """Give an accessor plot method the ``backend`` option.

    The decorated method declares ``backend=None`` in its signature (so that its docs and
    ``help()`` show it) and draws with Matplotlib as before; with ``backend="plotly"`` or
    ``backend="tikz"`` (or that default, see :func:`set_backend`) the decorator draws off screen
    and returns the converted result instead (see :mod:`plasma_plots.tikz_backend`). An ``ax``
    given with either raises ``TypeError``, since such a figure cannot be drawn into a
    Matplotlib axes.

    Parameters
    ----------
    method : callable
        A method returning a ``PlotResult``, a ``FuncAnimation`` or an
        ``InteractiveSliceViewer``.

    Returns
    -------
    callable
        The method with the ``backend`` option.
    """
    signature = inspect.signature(method)
    if "backend" not in signature.parameters:
        raise TypeError(f"{method.__qualname__} must declare a backend parameter")

    @functools.wraps(method)
    def wrapper(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        backend = resolve_backend(arguments.get("backend"))
        token, depth = _drawing.set(True), _depth.set(_depth.get() + 1)
        try:
            if backend == "matplotlib":
                result = method(*args, **kwargs)
                if _collecting and _depth.get() == 1:  # a panel of plasma_plots.figure
                    _collecting[-1].append(result)
                return result
            ax = arguments.get("ax", (arguments.get("options") or {}).get("ax"))
            if ax is not None:
                raise TypeError(
                    f"ax= draws into a Matplotlib axes; it cannot be combined with backend={backend!r}"
                )
            if backend == "tikz":
                from .tikz_backend import _drawn_as_tikz

                return _drawn_as_tikz(lambda: method(*args, **kwargs))
            return _drawn_as_plotly(lambda: method(*args, **kwargs))
        finally:
            _drawing.reset(token)
            _depth.reset(depth)

    return wrapper


@contextlib.contextmanager
def _offscreen():
    """Draw without showing anything, and close every figure created meanwhile."""
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    try:
        with plt.ioff():
            yield
    finally:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)


def _drawn_as_plotly(draw):
    """Run ``draw`` off screen and convert what it returns."""
    from .mpi import SkippedPlot
    from .plotting import InteractiveSliceViewer, PlotResult

    _plotly()  # fail before drawing if plotly is missing
    with _offscreen():
        result = draw()
        if isinstance(result, SkippedPlot) or result is None:
            return result
        if isinstance(result, PlotResult):
            if _is_plotly(result.fig):
                return result
            return _plotly_result(to_plotly(result.fig), result)
        if isinstance(result, InteractiveSliceViewer):
            return viewer_to_plotly(result)
        if _is_animation(result):
            figure = animation_to_plotly(result)
            return PlotResult(figure, None, list(figure.data))
    raise TypeError(f"cannot convert a {type(result).__name__} to Plotly")


def _plotly_result(figure, result):
    from .plotting import PlotResult

    return PlotResult(
        figure, None, list(figure.data), list(result.fit_results), dict(result.data)
    )


def _is_plotly(figure) -> bool:
    return type(figure).__module__.startswith("plotly")


def _is_animation(obj) -> bool:
    from matplotlib.animation import FuncAnimation

    return isinstance(obj, FuncAnimation)


# ---------------------------------------------------------------------------------------------
# Text, colors and styles
# ---------------------------------------------------------------------------------------------
_SYMBOLS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "ϑ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "parallel": "∥",
    "perp": "⊥",
    "nabla": "∇",
    "partial": "∂",
    "infty": "∞",
    "cdot": "·",
    "times": "×",
    "pm": "±",
    "mp": "∓",
    "sqrt": "√",
    "approx": "≈",
    "sim": "∼",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "propto": "∝",
    "langle": "⟨",
    "rangle": "⟩",
    "degree": "°",
    "circ": "°",
    "ell": "ℓ",
    "hbar": "ℏ",
    "int": "∫",
    "sum": "∑",
    "rightarrow": "→",
    "to": "→",
    "leftarrow": "←",
    "prime": "′",
    "odot": "⊙",
    "otimes": "⊗",
    "vert": "|",
    "|": "‖",
}
_SPACES = {",": " ", ";": " ", ":": " ", "!": "", " ": " ", "quad": " ", "qquad": "  "}
_FONTS = (
    "mathrm",
    "mathit",
    "mathbf",
    "mathcal",
    "mathsf",
    "mathtt",
    "text",
    "textrm",
    "operatorname",
    "boldsymbol",
    "vec",
    "hat",
    "bar",
    "tilde",
    "dot",
    "overline",
    "left",
    "right",
)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _group(math: str, i: int) -> tuple[str, int]:
    """The argument starting at ``math[i]``: a ``{...}`` group or one character (or command)."""
    if i >= len(math):
        return "", i
    if math[i] == "{":
        depth, j = 1, i + 1
        while j < len(math) and depth:
            depth += {"{": 1, "}": -1}.get(math[j], 0)
            j += 1
        return math[i + 1 : j - 1], j
    if math[i] == "\\":
        match = re.match(r"\\([A-Za-z]+|.)", math[i:])
        return match.group(0), i + len(match.group(0))
    return math[i], i + 1


def _math(math: str) -> str:
    """Mathtext (the part between ``$``) as Plotly text: Unicode and <sub>/<sup>."""
    out, i = [], 0
    while i < len(math):
        char = math[i]
        if char in "_^":
            argument, i = _group(math, i + 1)
            tag = "sub" if char == "_" else "sup"
            out.append(f"<{tag}>{_math(argument)}</{tag}>")
        elif char == "\\":
            match = re.match(r"\\([A-Za-z]+|.)", math[i:])
            name = match.group(1)
            i += len(match.group(0))
            if name in _SYMBOLS:
                out.append(_SYMBOLS[name])
            elif name in _SPACES:
                out.append(_SPACES[name])
            elif name == "frac":
                numerator, i = _group(math, i)
                denominator, i = _group(math, i)
                out.append(f"{_math(numerator)}/{_math(denominator)}")
            elif name in _FONTS:
                if i < len(math) and math[i] == "{":
                    argument, i = _group(math, i)
                    out.append(_math(argument))
            else:
                out.append(_escape(name))
        elif char in "{}":
            i += 1
        else:
            out.append(_escape(char))
            i += 1
    return "".join(out)


def plotly_text(text) -> str:
    """Turn a Matplotlib label into Plotly text.

    Mathtext (``$\\omega$``, ``$p_0$``) becomes Unicode and <sub>/<sup>; other ``<``, ``>`` and
    ``&`` are escaped, newlines become <br>.

    Parameters
    ----------
    text : str or None
        The label.

    Returns
    -------
    str
        The Plotly text; ``""`` for ``None``.

    Examples
    --------
    >>> plotly_text(r"fit: $\\gamma$ = 0.1")
    'fit: γ = 0.1'
    >>> plotly_text("$p_0$")
    'p<sub>0</sub>'
    """
    if not text:
        return ""
    parts = re.split(r"(?<!\\)\$", str(text))
    if len(parts) % 2 == 0:  # an unmatched $: not mathtext
        parts = [str(text)]
    out = [
        _math(part) if i % 2 else _escape(part.replace(r"\$", "$"))
        for i, part in enumerate(parts)
    ]
    return "".join(out).replace("\n", "<br>")


def _hover_label(text) -> str:
    """A label for hover templates: without tags, and without braces a template would read."""
    return re.sub(r"<[^>]+>", "", plotly_text(text)).replace("%{", "% {")


def _rgba(color, alpha=None) -> str:
    from matplotlib.colors import to_rgba

    r, g, b, a = to_rgba(color)
    if alpha is not None:
        a = alpha
    return f"rgba({round(255 * r)}, {round(255 * g)}, {round(255 * b)}, {a:.3g})"


def _colorscale(cmap, n=64):
    """A Matplotlib colormap sampled as a Plotly colorscale."""
    return [[i / (n - 1), _rgba(cmap(i / (n - 1)))] for i in range(n)]


_DASHES = {
    "-": "solid",
    "solid": "solid",
    "--": "dash",
    "dashed": "dash",
    ":": "dot",
    "dotted": "dot",
    "-.": "dashdot",
    "dashdot": "dashdot",
}


def _dash_pattern(offset_and_pattern):
    """A Matplotlib ``(offset, on-off sequence)`` in points as a Plotly dash in px."""
    if offset_and_pattern is None:
        return "solid"
    _, pattern = offset_and_pattern
    if not pattern:
        return "solid"
    return ",".join(f"{max(float(v) * PX_PER_PT, 0.5):.3g}px" for v in pattern)


def _line_dash(line):
    style = line.get_linestyle()
    if style in ("-", "solid"):
        return "solid"
    pattern = getattr(line, "_dash_pattern", None)
    if pattern is not None and pattern[1]:
        return _dash_pattern(pattern)
    return _DASHES.get(style, "solid")


_MARKERS = {
    "o": "circle",
    ".": "circle",
    ",": "square",
    "v": "triangle-down",
    "^": "triangle-up",
    "<": "triangle-left",
    ">": "triangle-right",
    "s": "square",
    "D": "diamond",
    "d": "diamond-tall",
    "x": "x-thin",
    "X": "x",
    "+": "cross-thin",
    "P": "cross",
    "*": "star",
    "p": "pentagon",
    "h": "hexagon",
    "H": "hexagon2",
    "8": "octagon",
    "|": "line-ns",
    "_": "line-ew",
    "1": "y-down",
    "2": "y-up",
    "3": "y-left",
    "4": "y-right",
}
_LINE_MARKERS = {
    "x-thin",
    "cross-thin",
    "line-ns",
    "line-ew",
    "y-down",
    "y-up",
    "y-left",
    "y-right",
}


@functools.lru_cache(maxsize=None)
def _marker_paths():
    from matplotlib.markers import MarkerStyle

    paths = []
    for name in _MARKERS:
        style = MarkerStyle(name)
        path = style.get_path().transformed(style.get_transform())
        paths.append((name, path.vertices, style.is_filled()))
    return paths


def _collection_marker(collection):
    """The Matplotlib marker of a scatter collection, found by comparing its path."""
    paths = collection.get_paths()
    if not paths:
        return "o"
    vertices = paths[0].vertices
    for name, candidate, _ in _marker_paths():
        if candidate.shape == vertices.shape and np.allclose(
            candidate, vertices, atol=1e-6
        ):
            return name
    return "o"


def _values(array):
    """A float array with masked or invalid values as NaN."""
    return np.ma.filled(np.ma.asarray(array, dtype=float), np.nan)


def _nan_joined(pieces):
    """Concatenate 1-D arrays with NaN between them, so Plotly draws them as separate lines."""
    pieces = [np.asarray(p, dtype=float) for p in pieces if len(p)]
    if not pieces:
        return np.array([], dtype=float)
    out = []
    for piece in pieces:
        out += [piece, [np.nan]]
    return np.concatenate(out[:-1])


# ---------------------------------------------------------------------------------------------
# Converting one figure
# ---------------------------------------------------------------------------------------------
def to_plotly(figure, *, strict: bool = False):
    """Convert a drawn Matplotlib figure into an interactive Plotly figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        A figure drawn by one of the plotting functions (or any figure with the artists they
        use, see :mod:`plasma_plots.plotly_backend`).
    strict : bool, optional
        Raise instead of warning when a part of the figure cannot be converted. Default:
        ``False``.

    Returns
    -------
    plotly.graph_objects.Figure
        The same content: traces for the data, shapes for lines and bands spanning an axes,
        annotations for text, a colorbar per Matplotlib colorbar.

    Raises
    ------
    ImportError
        If plotly is not installed.

    See Also
    --------
    animation_to_plotly : The same for an animation.
    plasma_plots.plotting.PlotResult.to_plotly : The same for a plot result.

    Examples
    --------
    >>> result = plot_slice(phi.isel(t=-1, eta3=0))
    >>> to_plotly(result.fig).write_html("phi.html")
    """
    go = _plotly()
    with warnings.catch_warnings():
        if strict:
            warnings.simplefilter("error", ConversionWarning)
        converted = _FigureConverter(figure).convert()
    return go.Figure(data=converted["data"], layout=converted["layout"])


class _Axes:
    """How one Matplotlib axes appears in the Plotly figure."""

    def __init__(self, ax, index, kind="2d", parent=None):
        self.ax, self.index, self.kind, self.parent = ax, index, kind, parent
        suffix = "" if index == 1 else str(index)
        if kind == "3d":
            self.scene = f"scene{suffix}"
        self.xaxis = parent.xaxis if parent is not None else f"xaxis{suffix}"
        self.yaxis = f"yaxis{suffix}"
        self.legend = None  # the Plotly legend name, if this axes has a legend

    @property
    def xref(self):
        return self.xaxis.replace("axis", "")

    @property
    def yref(self):
        return self.yaxis.replace("axis", "")


class _FigureConverter:
    def __init__(self, figure):
        self.figure = figure
        figure.draw_without_rendering()  # runs the layout engine; quivers autoscale here too
        width, height = figure.get_size_inches()
        self.width, self.height = width * PX_PER_INCH, height * PX_PER_INCH
        self.data, self.shapes, self.annotations, self.images = [], [], [], []
        self.layout = {}
        self.coloraxes = []  # (key, mappable, colorbar or None)

    # --- the layout ---------------------------------------------------------------------------
    def _classify(self):
        axes, colorbars = [], []
        for ax in self.figure.axes:
            if getattr(ax, "_colorbar", None) is not None:
                colorbars.append(ax._colorbar)
            elif ax.get_visible() and ax.get_navigate():
                axes.append(ax)
        converted, index = [], 0
        for ax in axes:
            twin_of = next(
                (
                    c
                    for c in converted
                    if c.kind == "2d"
                    and c.parent is None
                    and c.ax.get_shared_x_axes().joined(c.ax, ax)
                    and np.allclose(
                        c.ax.get_position().bounds, ax.get_position().bounds
                    )
                ),
                None,
            )
            index += 1
            kind = "3d" if ax.name == "3d" else "2d"
            converted.append(_Axes(ax, index, kind, parent=twin_of))
        return converted, colorbars

    def _paper(self, bounds):
        """Figure fractions ``(x0, y0, w, h)`` as Plotly paper fractions inside the margins."""
        m = self.margin
        x0, y0, w, h = bounds
        left = (x0 * self.width - m["l"]) / (self.width - m["l"] - m["r"])
        right = ((x0 + w) * self.width - m["l"]) / (self.width - m["l"] - m["r"])
        bottom = (y0 * self.height - m["b"]) / (self.height - m["t"] - m["b"])
        top = ((y0 + h) * self.height - m["b"]) / (self.height - m["t"] - m["b"])
        return [left, right], [bottom, top]

    def convert(self):
        self.axes, colorbars = self._classify()
        self.native = (
            sum(1 for a in self.axes if a.parent is None) == 1
            and self.axes[0].kind == "2d"
            if self.axes
            else True
        )
        self.margin = (
            {"l": 70, "r": 30, "t": 60, "b": 55}
            if self.native
            else {"l": 8, "r": 8, "t": 8, "b": 8}
        )
        # left to right, as Matplotlib placed them (Plotly stacks a lone axes' colorbars in this order)
        for colorbar in sorted(colorbars, key=lambda c: c.ax.get_position().x0):
            self._coloraxis(colorbar.mappable, colorbar)
        suptitle = (
            self.figure.get_suptitle() if hasattr(self.figure, "get_suptitle") else ""
        )
        for axes in self.axes:
            self._axes_layout(axes)
            self._artists(axes)
        self._coloraxis_layout()
        self._titles(suptitle)
        layout = self.layout
        layout.update(
            width=round(self.width),
            height=round(self.height),
            margin=self.margin,
            template="plotly_white",
            plot_bgcolor="white",
            hovermode="closest",
            showlegend=any(t.get("showlegend") for t in self.data)
            or any(s.get("showlegend") for s in self.shapes),
            meta={"converted_from": "matplotlib"},
        )
        if self.shapes:
            layout["shapes"] = self.shapes
        if self.annotations:
            layout["annotations"] = self.annotations
        if self.images:
            layout["images"] = self.images
        return {"data": self.data, "layout": layout}

    def _axes_layout(self, axes):
        ax = axes.ax
        if axes.kind == "3d":
            domain_x, domain_y = (
                self._paper(ax.get_position().bounds)
                if not self.native
                else ([0, 1], [0, 1])
            )
            self.layout[axes.scene] = {
                "domain": {"x": _clip(domain_x), "y": _clip(domain_y)},
                "xaxis": {"title": {"text": plotly_text(ax.get_xlabel())}},
                "yaxis": {"title": {"text": plotly_text(ax.get_ylabel())}},
                "zaxis": {"title": {"text": plotly_text(ax.get_zlabel())}},
                "aspectmode": "cube",  # Matplotlib's 3-D box, whatever the data's proportions
            }
            return
        common = dict(
            showline=True,
            mirror=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            zeroline=False,
        )
        y = dict(
            common, title={"text": plotly_text(ax.get_ylabel())}, **self._scale(ax, "y")
        )
        y["showgrid"] = _grid(ax.yaxis)
        if (
            axes.parent is not None
        ):  # a twin: its own y axis on the right, over the parent's
            y.update(
                overlaying=axes.parent.yref,
                side="right",
                anchor=axes.xref,
                mirror=False,
                showgrid=False,
                tickmode="auto",
            )  # its own ticks, not synced to the parent's grid
            self.layout[axes.yaxis] = y
            return
        x = dict(
            common, title={"text": plotly_text(ax.get_xlabel())}, **self._scale(ax, "x")
        )
        x["showgrid"] = _grid(ax.xaxis)
        twin = any(a.parent is axes for a in self.axes)
        if twin:
            x["mirror"] = False
            y["mirror"] = False
        x["anchor"], y["anchor"] = axes.yref, axes.xref
        if not self.native:
            domain_x, domain_y = self._paper(ax.get_position().bounds)
            x["domain"], y["domain"] = _clip(domain_x), _clip(domain_y)
        if ax.get_aspect() == 1.0:
            y.update(scaleanchor=axes.xref, scaleratio=1)
            x["constrain"] = y["constrain"] = "domain"
        # sharex/sharey: zooming one panel zooms the others
        for other in self.axes:
            if (
                other.index >= axes.index
                or other.parent is not None
                or other.kind != "2d"
            ):
                continue
            if "matches" not in x and ax.get_shared_x_axes().joined(ax, other.ax):
                x["matches"] = other.xref
            if "matches" not in y and ax.get_shared_y_axes().joined(ax, other.ax):
                y["matches"] = other.yref
        # a shared axis without tick labels (e.g. the upper panels of sharex=True) keeps them off
        if not any(label.get_visible() for label in ax.get_xticklabels()):
            x["showticklabels"] = False
        if not any(label.get_visible() for label in ax.get_yticklabels()):
            y["showticklabels"] = False
        self.layout[axes.xaxis], self.layout[axes.yaxis] = x, y

    @staticmethod
    def _scale(ax, which):
        from matplotlib.ticker import FixedLocator

        scale = ax.get_xscale() if which == "x" else ax.get_yscale()
        lo, hi = ax.get_xlim() if which == "x" else ax.get_ylim()
        out = {"exponentformat": "power"}  # 10⁻⁶, not Plotly's SI prefixes (1µ)
        if scale == "log":
            out["type"] = "log"
            if lo > 0 and hi > 0:
                out["range"] = [float(np.log10(lo)), float(np.log10(hi))]
                if (
                    abs(out["range"][1] - out["range"][0]) >= 1
                ):  # label the decades only, as Matplotlib
                    out["dtick"] = 1
        else:
            if scale != "linear":
                warnings.warn(
                    f"a {scale!r} axis scale is drawn linear in Plotly",
                    ConversionWarning,
                    stacklevel=4,
                )
            out["range"] = [float(lo), float(hi)]
        axis = ax.xaxis if which == "x" else ax.yaxis
        if isinstance(axis.get_major_locator(), FixedLocator):
            out["tickvals"] = [float(v) for v in axis.get_major_locator().locs]
        return out

    def _titles(self, suptitle):
        title = ""
        main = [a for a in self.axes if a.parent is None]
        if self.native and main:
            title = main[0].ax.get_title() or main[0].ax.get_title("left")
            if suptitle:
                self.layout["title"] = {
                    "text": plotly_text(title) or plotly_text(suptitle),
                    "x": 0.5,
                    "xanchor": "center",
                }
                if title:
                    self.layout["title"]["subtitle"] = {"text": plotly_text(suptitle)}
            elif title:
                self.layout["title"] = {
                    "text": plotly_text(title),
                    "x": 0.5,
                    "xanchor": "center",
                }
            return
        for axes in main:
            text = axes.ax.get_title() or axes.ax.get_title("left")
            if not text:
                continue
            if axes.kind == "3d":
                domain = self.layout[axes.scene]["domain"]
                x, y = sum(domain["x"]) / 2, domain["y"][1]
            else:
                x = sum(self.layout[axes.xaxis]["domain"]) / 2
                y = self.layout[axes.yaxis]["domain"][1]
            self.annotations.append(
                dict(
                    text=plotly_text(text),
                    x=x,
                    y=y,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="bottom",
                    yshift=4,
                    showarrow=False,
                    font={"size": 13},
                )
            )
        if suptitle:  # Matplotlib's layout left room for it at the top
            self.layout["title"] = {
                "text": plotly_text(suptitle),
                "x": 0.5,
                "xanchor": "center",
                "y": 0.995,
                "yanchor": "top",
                "yref": "container",
                "font": {"size": 14},
            }

    # --- colors -------------------------------------------------------------------------------
    def _coloraxis(self, mappable, colorbar=None):
        """The Plotly coloraxis of a mappable: shared by every mappable with the same colormap and
        limits as one with a colorbar (e.g. the panels of one shared colorbar)."""
        norm, cmap = mappable.norm, mappable.get_cmap()
        key = (
            getattr(cmap, "name", id(cmap)),
            type(norm).__name__,
            _float(norm.vmin),
            _float(norm.vmax),
        )
        for name, other, _, _ in self.coloraxes:  # its own colorbar's first
            if other is mappable:
                return name
        for name, _, other_colorbar, other_key in self.coloraxes:
            if colorbar is None and other_key == key and other_colorbar is not None:
                return name
        name = (
            "coloraxis" if not self.coloraxes else f"coloraxis{len(self.coloraxes) + 1}"
        )
        self.coloraxes.append((name, mappable, colorbar, key))
        return name

    def _coloraxis_layout(self):
        from matplotlib.colors import Normalize

        for position, (name, mappable, colorbar, _) in enumerate(self.coloraxes):
            norm = mappable.norm
            if type(norm) is not Normalize:
                warnings.warn(
                    f"a {type(norm).__name__} color scale is drawn linear",
                    ConversionWarning,
                    stacklevel=3,
                )
            axis = {
                "colorscale": _colorscale(mappable.get_cmap()),
                "showscale": colorbar is not None,
            }
            if norm.vmin is not None and np.isfinite(norm.vmin):
                axis["cmin"] = float(norm.vmin)
            if norm.vmax is not None and np.isfinite(norm.vmax):
                axis["cmax"] = float(norm.vmax)
            if colorbar is not None:
                horizontal = colorbar.orientation == "horizontal"
                label = (
                    colorbar.ax.get_xlabel() if horizontal else colorbar.ax.get_ylabel()
                )
                bar = {
                    "title": {"text": plotly_text(label), "side": "right"},
                    "thickness": 14,
                    "exponentformat": "power",
                }
                if not self.native:
                    xs, ys = self._paper(colorbar.ax.get_position().bounds)
                    bar.update(
                        x=xs[0],
                        xanchor="left",
                        y=sum(ys) / 2,
                        yanchor="middle",
                        len=ys[1] - ys[0],
                        lenmode="fraction",
                        xref="paper",
                        yref="paper",
                    )
                else:
                    bar.update(
                        x=1.02
                        + 0.14
                        * sum(1 for c in self.coloraxes[:position] if c[2] is not None)
                    )
                axis["colorbar"] = bar
            self.layout[name] = axis

    # --- artists ------------------------------------------------------------------------------
    def _artists(self, axes):
        from matplotlib.collections import (LineCollection, PathCollection,
                                            QuadMesh)
        from matplotlib.contour import ContourSet
        from matplotlib.legend import Legend
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib.quiver import Quiver
        from matplotlib.text import Annotation, Text

        ax = axes.ax
        legend = ax.get_legend()
        self._legend_labels = (
            {t.get_text() for t in legend.get_texts()} if legend is not None else set()
        )
        if legend is not None and self._legend_labels:
            self._legend(axes)
        skip = {
            ax.patch,
            ax.title,
            ax._left_title,
            ax._right_title,
            legend,
            *ax.spines.values(),
            ax.xaxis,
            ax.yaxis,
        }
        if axes.kind == "3d":
            skip |= {ax.zaxis}
        children = [c for c in ax.get_children() if c not in skip and c.get_visible()]
        children.sort(
            key=lambda c: c.get_zorder()
        )  # stable: equal zorders keep their order
        self._bins = {}  # colored line collections of this axes, pooled per coloraxis
        for artist in children:
            if axes.kind == "3d":
                self._artist_3d(axes, artist)
            elif isinstance(artist, QuadMesh):
                self._mesh(axes, artist)
            elif isinstance(artist, ContourSet):
                self._contours(axes, artist)
            elif isinstance(artist, Quiver):
                self._quiver(axes, artist)
            elif isinstance(artist, Line2D):
                self._line(axes, artist)
            elif isinstance(artist, PathCollection):
                self._scatter(axes, artist)
            elif isinstance(artist, LineCollection):
                self._line_collection(axes, artist)
            elif isinstance(artist, Patch):
                self._patch(axes, artist)
            elif isinstance(artist, Annotation):
                self._text(axes, artist)
            elif isinstance(artist, Text):
                if artist.get_text():
                    self._text(axes, artist)
            elif isinstance(artist, Legend):
                continue
            else:
                warnings.warn(
                    f"a {type(artist).__name__} is not converted to Plotly",
                    ConversionWarning,
                    stacklevel=3,
                )
        self._flush_bins(axes)

    def _legend(self, axes):
        """One Plotly legend per Matplotlib legend: beside a lone axes, else inside its axes."""
        owner = axes.parent or axes
        if owner.legend is not None:  # a twin shares its parent's legend
            axes.legend = owner.legend
            return
        count = sum(1 for a in self.axes if a.parent is None and a.legend is not None)
        name = "legend" if count == 0 else f"legend{count + 1}"
        owner.legend = axes.legend = name
        colorbars = any(colorbar is not None for _, _, colorbar, _ in self.coloraxes)
        twins = any(a.parent is owner for a in self.axes)
        if self.native and not colorbars and not twins:
            self.layout[name] = {
                "bgcolor": "rgba(255, 255, 255, 0.8)"
            }  # Plotly's place, right of the axes
            return
        if self.native:
            x, y = 0.99, 0.99
        elif owner.kind == "3d":
            domain_x, domain_y = self._paper(owner.ax.get_position().bounds)
            x, y = _clip(domain_x)[1], _clip(domain_y)[1]
        else:
            domain_x, domain_y = self._paper(owner.ax.get_position().bounds)
            x, y = _clip(domain_x)[1] - 0.005, _clip(domain_y)[1] - 0.005
        self.layout[name] = {
            "x": x,
            "y": y,
            "xanchor": "right",
            "yanchor": "top",
            "xref": "paper",
            "yref": "paper",
            "bgcolor": "rgba(255, 255, 255, 0.75)",
            "bordercolor": "rgba(0, 0, 0, 0.2)",
            "borderwidth": 1,
            "font": {"size": 10},
        }

    def _shown(self, axes, label) -> dict:
        """The legend options of a trace with this Matplotlib label."""
        label = label or ""
        visible = (
            bool(label) and not label.startswith("_") and label in self._legend_labels
        )
        out = {
            "name": plotly_text(label) if label and not label.startswith("_") else "",
            "showlegend": visible,
        }
        if visible and axes.legend not in (None, "legend"):
            out["legend"] = axes.legend
        return out

    def _refs(self, axes):
        return {"xaxis": axes.xref, "yaxis": axes.yref}

    def _branches(self, axes, artist):
        """Whether x and y of an artist are in data coordinates (else in axes fractions)."""
        return artist.get_transform().contains_branch_seperately(axes.ax.transData)

    def _line(self, axes, line):
        x_data, y_data = self._branches(axes, line)
        color = _rgba(line.get_color(), line.get_alpha())
        width = line.get_linewidth() * PX_PER_PT
        dash = _line_dash(line)
        shown = self._shown(axes, line.get_label())
        if not (x_data and y_data):
            if not (x_data or y_data):
                warnings.warn(
                    "a line in axes coordinates is not converted",
                    ConversionWarning,
                    stacklevel=4,
                )
                return
            # axhline / axvline: a shape spanning the axes, whatever the zoom
            xs, ys = np.asarray(line.get_xdata(), float), np.asarray(
                line.get_ydata(), float
            )
            if y_data:  # horizontal
                shape = dict(
                    type="line",
                    xref=f"{axes.xref} domain",
                    yref=axes.yref,
                    x0=float(xs.min()),
                    x1=float(xs.max()),
                    y0=float(ys[0]),
                    y1=float(ys[-1]),
                )
            else:
                shape = dict(
                    type="line",
                    xref=axes.xref,
                    yref=f"{axes.yref} domain",
                    x0=float(xs[0]),
                    x1=float(xs[-1]),
                    y0=float(ys.min()),
                    y1=float(ys.max()),
                )
            shape["line"] = {"color": color, "width": width, "dash": dash}
            shape["layer"] = "above"
            if shown["showlegend"]:
                shape.update(
                    showlegend=True,
                    name=shown["name"],
                    **({"legend": shown["legend"]} if "legend" in shown else {}),
                )
            self.shapes.append(shape)
            return
        trace = self._line_style(line, color, width, dash)
        trace.update(
            type="scatter",
            x=_values(line.get_xdata()),
            y=_values(line.get_ydata()),
            **shown,
            **self._refs(axes),
        )
        self.data.append(trace)

    @staticmethod
    def _line_style(line, color, width, dash):
        linestyle = line.get_linestyle()
        has_line = (
            linestyle not in ("None", "", " ", "none") and line.get_linewidth() > 0
        )
        marker = line.get_marker()
        has_marker = marker not in (None, "None", "", " ", "none")
        mode = (
            "+".join(
                m for m, on in (("lines", has_line), ("markers", has_marker)) if on
            )
            or "lines"
        )
        trace = {"mode": mode}
        if has_line:
            trace["line"] = {"color": color, "width": width, "dash": dash}
        if has_marker:
            symbol = (
                _MARKERS.get(marker, "circle") if isinstance(marker, str) else "circle"
            )
            size = line.get_markersize() * PX_PER_PT * (0.5 if marker == "." else 1.0)
            face = line.get_markerfacecolor()
            edge = _rgba(line.get_markeredgecolor(), line.get_alpha())
            open_face = isinstance(face, str) and face.lower() == "none"
            if symbol in _LINE_MARKERS:
                trace["marker"] = {
                    "symbol": symbol,
                    "size": size,
                    "color": edge,
                    "line": {
                        "color": edge,
                        "width": max(line.get_markeredgewidth() * PX_PER_PT, 1),
                    },
                }
            elif open_face:
                trace["marker"] = {
                    "symbol": f"{symbol}-open",
                    "size": size,
                    "color": edge,
                    "line": {
                        "color": edge,
                        "width": max(line.get_markeredgewidth() * PX_PER_PT, 1),
                    },
                }
            else:
                trace["marker"] = {
                    "symbol": symbol,
                    "size": size,
                    "color": _rgba(face, line.get_alpha()),
                    "line": {
                        "color": edge,
                        "width": line.get_markeredgewidth() * PX_PER_PT * 0.5,
                    },
                }
        return trace

    def _mesh(self, axes, mesh):
        coordinates = np.asarray(
            mesh.get_coordinates(), dtype=float
        )  # (M + 1, N + 1, 2) corners
        values = _values(mesh.get_array())
        if values.ndim == 1:
            values = values.reshape(coordinates.shape[0] - 1, coordinates.shape[1] - 1)
        alpha = mesh.get_alpha()
        coloraxis = self._coloraxis(mesh)
        if (
            alpha is not None and alpha == 0
        ):  # fill=False: only its colorbar (and contour lines) show
            self._colorbar_holder(axes, coloraxis)
            return
        X, Y = coordinates[..., 0], coordinates[..., 1]
        hover = self._hover_template(axes, coloraxis, value="%{z:.4g}")
        if np.allclose(X, X[:, :1]) and np.allclose(
            Y, Y[:1, :]
        ):  # x along rows: the "ij" grids of plot_slice
            x_edges, y_edges, z = X[:, 0], Y[0, :], values.T
        elif np.allclose(X, X[:1, :]) and np.allclose(Y, Y[:, :1]):
            x_edges, y_edges, z = X[0, :], Y[:, 0], values
        else:
            self._curvilinear_mesh(axes, mesh, X, Y, values, coloraxis)
            return
        trace = dict(
            type="heatmap",
            x=x_edges,
            y=y_edges,
            z=z,
            coloraxis=coloraxis,
            hovertemplate=hover,
            zsmooth=False,
            **self._refs(axes),
        )
        if alpha is not None and alpha < 1:
            trace["opacity"] = float(alpha)
        self.data.append(trace)

    def _hover_template(self, axes, coloraxis, value):
        ax = axes.ax
        colorbar = next(
            (c for name, _, c, _ in self.coloraxes if name == coloraxis), None
        )
        label = ""
        if colorbar is not None:
            label = (
                colorbar.ax.get_xlabel()
                if colorbar.orientation == "horizontal"
                else colorbar.ax.get_ylabel()
            )
        xlabel, ylabel = (
            _hover_label(ax.get_xlabel()) or "x",
            _hover_label(ax.get_ylabel()) or "y",
        )
        return f"{xlabel} = %{{x:.4g}}<br>{ylabel} = %{{y:.4g}}<br>{_hover_label(label) or 'value'} = {value}<extra></extra>"

    def _curvilinear_mesh(self, axes, mesh, X, Y, values, coloraxis):
        """A mesh on a mapped grid: an image of it rendered by Matplotlib, under invisible markers at
        the cell centers that show the values on hover and carry the colorbar."""
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        ax = axes.ax
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        box = ax.get_window_extent()
        scale = min(IMAGE_PIXELS / max(box.width, box.height, 1.0), 3.0)
        width, height = max(box.width * scale, 8.0), max(box.height * scale, 8.0)
        raster = Figure(figsize=(width / 100.0, height / 100.0), dpi=100)
        FigureCanvasAgg(raster)
        target = raster.add_axes((0, 0, 1, 1))
        target.set_axis_off()
        target.pcolormesh(
            X,
            Y,
            np.ma.masked_invalid(values),
            cmap=mesh.get_cmap(),
            norm=mesh.norm,
            shading="flat",
            antialiased=False,
            alpha=mesh.get_alpha(),
        )
        target.set_xlim(x0, x1)
        target.set_ylim(y0, y1)
        buffer = io.BytesIO()
        raster.savefig(buffer, format="png", transparent=True, dpi=100)
        source = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )
        self.images.append(
            dict(
                source=source,
                xref=axes.xref,
                yref=axes.yref,
                x=min(x0, x1),
                y=max(y0, y1),
                sizex=abs(x1 - x0),
                sizey=abs(y1 - y0),
                xanchor="left",
                yanchor="top",
                sizing="stretch",
                layer="below",
            )
        )
        cx = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[:-1, 1:] + X[1:, 1:])
        cy = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:] + Y[1:, 1:])
        keep = np.isfinite(values)
        self.data.append(
            dict(
                type="scattergl" if keep.sum() > 20000 else "scatter",
                mode="markers",
                x=cx[keep],
                y=cy[keep],
                marker={
                    "color": values[keep].astype(np.float32),
                    "coloraxis": coloraxis,
                    "opacity": 0,
                    "size": 6,
                },
                hovertemplate=self._hover_template(
                    axes, coloraxis, value="%{marker.color:.4g}"
                ),
                showlegend=False,
                name="",
                **self._refs(axes),
            )
        )

    def _colorbar_holder(self, axes, coloraxis):
        """An invisible trace that shows a coloraxis' colorbar when no visible trace uses it."""
        self.data.append(
            dict(
                type="scatter",
                x=[None],
                y=[None],
                mode="markers",
                showlegend=False,
                name="",
                hoverinfo="skip",
                marker={"color": [0], "coloraxis": coloraxis, "opacity": 0},
                **self._refs(axes),
            )
        )

    def _contours(self, axes, contours):
        if contours.filled:
            warnings.warn(
                "filled contours are not converted", ConversionWarning, stacklevel=4
            )
            return
        colors = contours.get_edgecolor()
        widths = np.atleast_1d(contours.get_linewidth())
        styles = contours.get_linestyle()
        paths = contours.get_paths()
        for i, (level, path) in enumerate(zip(contours.levels, paths)):
            xs, ys = [], []
            if len(path.vertices):
                for segment in path.to_polygons(closed_only=False):
                    xs.append(segment[:, 0])
                    ys.append(segment[:, 1])
            color = _rgba(colors[i % len(colors)]) if len(colors) else "black"
            style = styles[i % len(styles)] if len(styles) else None
            dash = (
                _dash_pattern(style)
                if isinstance(style, tuple)
                else _DASHES.get(style, "solid")
            )
            self.data.append(
                dict(
                    type="scatter",
                    mode="lines",
                    x=_nan_joined(xs).astype(np.float32),
                    y=_nan_joined(ys).astype(np.float32),
                    showlegend=False,
                    name=f"level {level:.4g}",
                    line={
                        "color": color,
                        "width": widths[i % len(widths)] * PX_PER_PT,
                        "dash": dash,
                    },
                    hovertemplate=f"level = {level:.4g}<extra></extra>",
                    **self._refs(axes),
                )
            )

    def _scatter(self, axes, collection):
        if not all(
            collection.get_offset_transform().contains_branch_seperately(
                axes.ax.transData
            )
        ):
            warnings.warn(
                "a scatter outside data coordinates is not converted",
                ConversionWarning,
                stacklevel=4,
            )
            return
        offsets = np.asarray(collection.get_offsets(), dtype=float)
        marker = self._scatter_marker(axes, collection)
        trace = dict(
            type="scattergl" if len(offsets) > 20000 else "scatter",
            mode="markers",
            x=offsets[:, 0] if len(offsets) else [None],
            y=(
                offsets[:, 1] if len(offsets) else [None]
            ),  # a legend-only entry needs a point
            marker=marker,
            **self._shown(axes, collection.get_label()),
            **self._refs(axes),
        )
        self.data.append(trace)

    def _scatter_marker(self, axes, collection):
        symbol_name = _collection_marker(collection)
        symbol = _MARKERS[symbol_name]
        sizes = np.sqrt(np.asarray(collection.get_sizes(), dtype=float)) * PX_PER_PT
        size = float(sizes[0]) if sizes.size == 1 else sizes
        widths = np.atleast_1d(collection.get_linewidths())
        marker = {"symbol": symbol, "size": size}
        alpha = collection.get_alpha()
        array = collection.get_array()
        faces = collection.get_facecolors()
        edges = collection.get_edgecolors()
        if array is not None:
            marker.update(
                color=_values(array).ravel(), coloraxis=self._coloraxis(collection)
            )
            if alpha is not None:
                marker["opacity"] = float(alpha)
        else:
            colors = faces if len(faces) else edges
            marker["color"] = (
                _rgba(colors[0]) if len(colors) == 1 else [_rgba(c) for c in colors]
            )
        if symbol in _LINE_MARKERS:
            line_colors = edges if len(edges) else faces
            marker["line"] = {
                "color": (
                    _rgba(line_colors[0])
                    if len(line_colors)
                    else marker.get("color", "black")
                ),
                "width": max(float(widths[0]) * PX_PER_PT, 1.0),
            }
            if array is not None:
                marker["line"].pop("color")
        elif len(edges) and float(widths[0]) > 0:
            marker["line"] = {
                "color": _rgba(edges[0]),
                "width": float(widths[0]) * PX_PER_PT * 0.5,
            }
        else:
            marker["line"] = {"width": 0}
        return marker

    _BINS = 32

    def _line_collection(self, axes, collection):
        """Segments, colored by a value (e.g. orbits colored by time). A Plotly line has one color,
        so the segments of every such collection in the axes are pooled into 32 colors.
        """
        segments = collection.get_segments()
        array = collection.get_array()
        if array is None:
            colors = collection.get_colors()
            color = _rgba(colors[0]) if len(colors) else "black"
            self.data.append(
                dict(
                    type="scatter",
                    mode="lines",
                    x=_nan_joined([s[:, 0] for s in segments]),
                    y=_nan_joined([s[:, 1] for s in segments]),
                    line={"color": color},
                    **self._shown(axes, collection.get_label()),
                    **self._refs(axes),
                )
            )
            return
        values = _values(array).ravel()
        coloraxis = self._coloraxis(collection)
        pool = self._bins.setdefault(
            coloraxis,
            {
                "cmap": collection.get_cmap(),
                "width": float(np.atleast_1d(collection.get_linewidths())[0])
                * PX_PER_PT,
                "segments": [[] for _ in range(self._BINS)],
            },
        )
        scaled = np.clip(np.nan_to_num(collection.norm(values), nan=0.0), 0, 1)
        which = np.minimum((scaled * self._BINS).astype(int), self._BINS - 1)
        for i, b in enumerate(which):
            pool["segments"][b].append(segments[i])

    def _flush_bins(self, axes):
        for coloraxis, pool in self._bins.items():
            for b, chosen in enumerate(pool["segments"]):
                self.data.append(
                    dict(
                        type="scatter",
                        mode="lines",
                        x=_nan_joined([s[:, 0] for s in chosen]),
                        y=_nan_joined([s[:, 1] for s in chosen]),
                        line={
                            "color": _rgba(pool["cmap"]((b + 0.5) / self._BINS)),
                            "width": pool["width"],
                        },
                        showlegend=False,
                        name="",
                        hoverinfo="x+y",
                        **self._refs(axes),
                    )
                )
            self._colorbar_holder(axes, coloraxis)
        self._bins = {}

    def _quiver(self, axes, quiver):
        """Arrows drawn as in Matplotlib: lengths from its autoscaled ``scale``, directions in screen
        space (``angles="uv"``), converted to data units of the current view."""
        ax = axes.ax
        box = ax.get_window_extent()
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        X, Y = np.ravel(quiver.X), np.ravel(quiver.Y)
        U, V = _values(quiver.U).ravel(), _values(quiver.V).ravel()
        scale = quiver.scale or 1.0
        reference = box.width if quiver.units in ("width", None) else box.height
        du, dv = U / scale * reference, V / scale * reference  # screen pixels
        to_x, to_y = (x1 - x0) / box.width, (y1 - y0) / box.height
        tips_x, tips_y = X + du * to_x, Y + dv * to_y
        length = np.hypot(du, dv)
        angle = np.arctan2(dv, du)
        head = 0.3 * length
        xs, ys = [], []
        for side in (1, -1):
            a = angle + np.pi - side * np.radians(22)
            xs.append(tips_x + head * np.cos(a) * to_x)
            ys.append(tips_y + head * np.sin(a) * to_y)
        nan = np.full_like(X, np.nan)
        px = np.stack([X, tips_x, nan, xs[0], tips_x, xs[1], nan], axis=1).ravel()
        py = np.stack([Y, tips_y, nan, ys[0], tips_y, ys[1], nan], axis=1).ravel()
        color = quiver.get_facecolor()
        color = _rgba(color[0]) if len(color) else "black"
        self.data.append(
            dict(
                type="scatter",
                mode="lines",
                x=px,
                y=py,
                line={"color": color, "width": 1.2},
                hoverinfo="skip",
                **self._shown(axes, quiver.get_label()),
                **self._refs(axes),
            )
        )
        self.data.append(
            dict(
                type="scatter",
                mode="markers",
                x=X,
                y=Y,
                marker={"size": 6, "opacity": 0},
                customdata=np.stack([U, V], axis=1),
                showlegend=False,
                name="",
                hovertemplate="x = %{x:.4g}, y = %{y:.4g}<br>(%{customdata[0]:.4g}, "
                "%{customdata[1]:.4g})<extra></extra>",
                **self._refs(axes),
            )
        )

    def _patch(self, axes, patch):
        from matplotlib.patches import Polygon, Rectangle

        x_data, y_data = self._branches(axes, patch)
        face = patch.get_facecolor()
        fill = _rgba(face) if patch.get_fill() else "rgba(0, 0, 0, 0)"
        shown = self._shown(axes, patch.get_label())
        if isinstance(patch, Rectangle) and (x_data or y_data):
            x, y = patch.get_x(), patch.get_y()
            shape = dict(
                type="rect",
                xref=axes.xref if x_data else f"{axes.xref} domain",
                yref=axes.yref if y_data else f"{axes.yref} domain",
                x0=x,
                x1=x + patch.get_width(),
                y0=y,
                y1=y + patch.get_height(),
                fillcolor=fill,
                line={"width": 0},
                layer="below",
            )
            if shown["showlegend"]:
                shape.update(
                    showlegend=True,
                    name=shown["name"],
                    **({"legend": shown["legend"]} if "legend" in shown else {}),
                )
            self.shapes.append(shape)
            return
        if isinstance(patch, Polygon) and x_data and y_data:
            xy = np.asarray(patch.get_xy(), dtype=float)
            self.data.append(
                dict(
                    type="scatter",
                    mode="lines",
                    x=xy[:, 0],
                    y=xy[:, 1],
                    fill="toself",
                    fillcolor=fill,
                    line={"width": 0},
                    **shown,
                    **self._refs(axes),
                )
            )
            return
        warnings.warn(
            f"a {type(patch).__name__} is not converted to Plotly",
            ConversionWarning,
            stacklevel=4,
        )

    def _text(self, axes, text):
        from matplotlib.text import Annotation

        ax = axes.ax
        offset = (0.0, 0.0)
        if isinstance(text, Annotation):
            x, y = text.xy
            coords = {"data": (True, True), "axes fraction": (False, False)}
            if text.xycoords not in coords:
                warnings.warn(
                    f"an annotation at {text.xycoords!r} is not converted",
                    ConversionWarning,
                    stacklevel=4,
                )
                return
            x_data, y_data = coords[text.xycoords]
            if text.anncoords == "offset points":
                offset = np.asarray(text.xyann, dtype=float) * PX_PER_PT
            elif text.anncoords != text.xycoords or text.xyann != text.xy:
                warnings.warn(
                    f"annotation text at {text.anncoords!r} is placed at its point",
                    ConversionWarning,
                    stacklevel=4,
                )
        else:
            x, y = text.get_position()
            x_data, y_data = text.get_transform().contains_branch_seperately(
                ax.transData
            )
        yref = axes.yref
        ha = {"left": "left", "center": "center", "right": "right"}.get(
            text.get_horizontalalignment(), "left"
        )
        va = {
            "top": "top",
            "center": "middle",
            "center_baseline": "middle",
            "bottom": "bottom",
            "baseline": "bottom",
        }.get(text.get_verticalalignment(), "bottom")
        xscale, yscale = ax.get_xscale(), ax.get_yscale()
        xv = float(np.log10(x)) if x_data and xscale == "log" and x > 0 else float(x)
        yv = float(np.log10(y)) if y_data and yscale == "log" and y > 0 else float(y)
        self.annotations.append(
            dict(
                text=plotly_text(text.get_text()),
                x=xv,
                y=yv,
                xref=axes.xref if x_data else f"{axes.xref} domain",
                yref=yref if y_data else f"{yref} domain",
                xanchor=ha,
                yanchor=va,
                xshift=float(offset[0]),
                yshift=float(offset[1]),
                showarrow=False,
                font={
                    "size": text.get_fontsize() * PX_PER_PT * 0.95,
                    "color": _rgba(text.get_color()),
                },
            )
        )

    def _artist_3d(self, axes, artist):
        from mpl_toolkits.mplot3d.art3d import Line3D, Path3DCollection

        scene = axes.scene
        if isinstance(artist, Line3D):
            xs, ys, zs = (np.asarray(v, dtype=float) for v in artist.get_data_3d())
            self.data.append(
                dict(
                    type="scatter3d",
                    mode="lines",
                    x=xs,
                    y=ys,
                    z=zs,
                    scene=scene,
                    line={
                        "color": _rgba(artist.get_color(), artist.get_alpha()),
                        "width": max(artist.get_linewidth() * PX_PER_PT, 2.0),
                    },
                    **self._shown(axes, artist.get_label()),
                )
            )
        elif isinstance(artist, Path3DCollection):
            xs, ys, zs = (np.asarray(v, dtype=float) for v in artist._offsets3d)
            marker = self._scatter_marker(axes, artist)
            marker.pop("line", None)
            marker["size"] = (
                marker["size"] * 0.6
                if np.isscalar(marker["size"])
                else marker["size"] * 0.6
            )
            self.data.append(
                dict(
                    type="scatter3d",
                    mode="markers",
                    x=xs,
                    y=ys,
                    z=zs,
                    scene=scene,
                    marker=marker,
                    **self._shown(axes, artist.get_label()),
                )
            )
        elif getattr(artist, "get_text", None) is not None and not artist.get_text():
            return
        else:
            warnings.warn(
                f"a {type(artist).__name__} in a 3-D axes is not converted",
                ConversionWarning,
                stacklevel=4,
            )


def _grid(axis) -> bool:
    return any(line.get_visible() for line in axis.get_gridlines())


def _clip(interval):
    lo, hi = (float(min(max(v, 0.0), 1.0)) for v in interval)
    return [lo, max(hi, lo + 1e-3)]


def _float(value):
    return None if value is None else float(value)


# ---------------------------------------------------------------------------------------------
# Animations and viewers
# ---------------------------------------------------------------------------------------------
_FRAME_LAYOUT = ("title", "annotations", "images", "shapes")


def animation_to_plotly(
    animation,
    *,
    labels=None,
    prefix: str | None = None,
    play: bool = True,
    strict: bool = False,
):
    """Convert a Matplotlib animation of the plotting functions into a Plotly figure with frames.

    Every frame is drawn by the animation's own update function and converted like a figure, so
    the frames show exactly what the Matplotlib animation shows; a slider (and Play/Pause buttons)
    steps through them.

    Parameters
    ----------
    animation : matplotlib.animation.FuncAnimation
        The animation, e.g. from :func:`~plasma_plots.plotting.animate_slices`.
    labels : sequence of str, optional
        One slider label per frame. Default: the sweep values the animation was made for (for the
        animations of this package), else the frame numbers.
    prefix : str, optional
        Shown before the current label, e.g. ``"t = "``. Default: the sweep's name.
    play : bool, optional
        Add Play and Pause buttons. Default: ``True``.
    strict : bool, optional
        Raise instead of warning when a part of a frame cannot be converted. Default: ``False``.

    Returns
    -------
    plotly.graph_objects.Figure
        The first frame, with every frame in ``figure.frames``.

    Raises
    ------
    ValueError
        If the frames do not all have the same kinds of traces.

    See Also
    --------
    to_plotly : The conversion of one figure.

    Examples
    --------
    >>> animation = animate_slices(phi.isel(eta3=0), step=2)
    >>> animation_to_plotly(animation).write_html("phi.html")
    """
    go = _plotly()
    frames = list(animation.new_frame_seq())
    if not frames:
        raise ValueError("the animation has no frames")
    sweep = getattr(animation, "_plasma_sweep", None)
    if labels is None:
        labels = (
            [f"{v:.4g}" for v in sweep[1]]
            if sweep is not None and len(sweep[1]) == len(frames)
            else [str(i) for i in range(len(frames))]
        )
    if prefix is None:
        prefix = f"{sweep[0]} = " if sweep is not None else ""
    if len(labels) != len(frames):
        raise ValueError(f"got {len(labels)} labels for {len(frames)} frames")
    snapshots = []
    with warnings.catch_warnings():
        if strict:
            warnings.simplefilter("error", ConversionWarning)
        for frame in frames:
            animation._func(frame, *animation._args)
            snapshots.append(_FigureConverter(animation._fig).convert())
    animation._draw_was_started = (
        True  # it was drawn, frame by frame: no "deleted without rendering" warning
    )
    traces = _aligned(snapshots)  # one list per frame, the same traces in every one
    first = snapshots[0]
    figure = go.Figure(data=traces[0], layout=first["layout"])
    # a frame names only what changes: traces that are the same in every frame (a static contour,
    # the domain's boundary, a fixed background) and the coordinates every frame shares (a mesh's
    # grid) are stored once, in the figure itself
    static = [
        j
        for j in range(len(traces[0]))
        if all(_same(frame[j], traces[0][j]) for frame in traces[1:])
    ]
    changing = [j for j in range(len(traces[0])) if j not in static]
    constant = _constant_keys(traces)
    figure.frames = [
        go.Frame(
            data=[
                {k: v for k, v in frame[j].items() if (j, k) not in constant}
                for j in changing
            ],
            traces=changing,
            name=str(i),
            layout={
                key: snapshot["layout"].get(key, [] if key != "title" else {"text": ""})
                for key in _FRAME_LAYOUT
                + tuple(k for k in snapshot["layout"] if k.startswith("coloraxis"))
            },
        )
        for i, (frame, snapshot) in enumerate(zip(traces, snapshots))
    ]
    duration = int(getattr(animation, "_interval", 100))
    steps = [
        dict(
            method="animate",
            label=label,
            args=[
                [str(i)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        )
        for i, label in enumerate(labels)
    ]
    extra = 90 if play else 70
    figure.update_layout(
        height=figure.layout.height + extra,
        margin={
            **figure.layout.margin.to_plotly_json(),
            "b": figure.layout.margin.b + extra,
        },
        sliders=[
            dict(
                active=0,
                steps=steps,
                x=0.08 if play else 0.0,
                len=0.92 if play else 1.0,
                y=0,
                yanchor="top",
                pad={"t": 45 if play else 35},
                currentvalue={"prefix": prefix, "xanchor": "right"},
            )
        ],
    )
    if play:
        figure.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    showactive=False,
                    x=0.0,
                    y=0,
                    xanchor="left",
                    yanchor="top",
                    pad={"t": 50, "r": 10},
                    buttons=[
                        dict(
                            label="▶",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": duration, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="❚❚",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ]
        )
    return figure


def _signature(trace):
    return (
        trace.get("type"),
        trace.get("name", ""),
        trace.get("mode", ""),
        trace.get("xaxis"),
        trace.get("yaxis"),
    )


def _aligned(snapshots):
    """The traces of every frame, aligned: a trace missing from some frames (e.g. a contour level the
    field reaches only later) is kept in every frame, hidden where it is missing."""
    import difflib

    union = [_signature(t) for t in snapshots[0]["data"]]
    for snapshot in snapshots[1:]:
        signatures = [_signature(t) for t in snapshot["data"]]
        merged = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, union, signatures, autojunk=False
        ).get_opcodes():
            if tag == "equal":
                merged += union[i1:i2]
            else:  # keep both sides, in order: the union only grows
                merged += union[i1:i2] + signatures[j1:j2]
        union = merged
    frames, example = [], {}
    for snapshot in snapshots:
        signatures = [_signature(t) for t in snapshot["data"]]
        row, position = [None] * len(union), 0
        for trace, signature in zip(snapshot["data"], signatures):
            while union[position] != signature:
                position += 1
            row[position] = trace
            example.setdefault(position, trace)
            position += 1
        frames.append(row)
    sometimes = {j for row in frames for j, trace in enumerate(row) if trace is None}
    for row in frames:
        for j in sometimes:
            if row[j] is None:  # the same kind of trace, hidden, with nothing to show
                hidden = {
                    k: v
                    for k, v in example[j].items()
                    if k
                    in (
                        "type",
                        "name",
                        "mode",
                        "xaxis",
                        "yaxis",
                        "showlegend",
                        "legend",
                        "line",
                        "marker",
                    )
                }
                row[j] = {**hidden, "x": [], "y": [], "visible": False}
            else:
                row[j] = {**row[j], "visible": True}
    return frames


def _same(a, b) -> bool:
    """Whether two converted traces (nested dicts, lists and arrays) are equal."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (np.ndarray, list, tuple)) or isinstance(
        b, (np.ndarray, list, tuple)
    ):
        try:
            x, y = np.asarray(a), np.asarray(b)
        except ValueError:
            return False
        if x.shape != y.shape:
            return False
        if x.dtype.kind in "fc" or y.dtype.kind in "fc":
            return bool(
                np.array_equal(x.astype(float), y.astype(float), equal_nan=True)
            )
        return bool(np.array_equal(x, y))
    return a == b


def _constant_keys(frames):
    """``(trace index, key)`` of the coordinates that are the same in every frame."""
    constant = set()
    for i, trace in enumerate(frames[0]):
        for key in ("x", "y", "z"):
            if key in trace and all(
                key in frame[i] and _same(frame[i][key], trace[key])
                for frame in frames[1:]
            ):
                constant.add((i, key))
    return constant


def viewer_to_plotly(viewer):
    """Convert a slider viewer into a Plotly figure with one slider.

    The Matplotlib viewer has one slider per remaining dimension; a Plotly figure can combine only
    one, so exactly one dimension besides the two drawn may be left with more than one value.

    Parameters
    ----------
    viewer : plasma_plots.plotting.InteractiveSliceViewer
        The viewer (drawn or not).

    Returns
    -------
    PlotResult
        The Plotly figure, with a slider over the remaining dimension (a static slice if there is
        none).

    Raises
    ------
    ValueError
        If more than one dimension remains to slide over.

    See Also
    --------
    animation_to_plotly : The conversion behind this one.
    """
    from .plotting import PlotResult, View, _select, animate_slices, plot_slice

    data = _select(viewer.data, viewer.view)
    x, y = viewer.view.x, viewer.view.y
    if x is None or y is None:
        candidates = [dim for dim in data.dims if dim != viewer.view.sweep]
        if len(candidates) < 2:
            raise ValueError("viewer needs two display dimensions")
        x, y = candidates[:2]
    controls = [dim for dim in data.dims if dim not in (x, y)]
    single = [dim for dim in controls if data.sizes[dim] == 1]
    data = data.isel({dim: 0 for dim in single})
    controls = [dim for dim in controls if dim not in single]
    if len(controls) > 1:
        raise ValueError(
            f"a Plotly viewer has one slider, but {controls} remain; select all but one of them (e.g. {controls[-1]}=0)"
        )
    base = View(x=x, y=y, coordinates=viewer.view.coordinates, plane=viewer.view.plane)
    options = dict(viewer.options)
    if not controls:
        result = plot_slice(data, view=base, **options)
        return _plotly_result(to_plotly(result.fig), result)
    sweep = controls[0]
    view = View(
        x=x,
        y=y,
        sweep=sweep,
        coordinates=viewer.view.coordinates,
        plane=viewer.view.plane,
    )
    animation = animate_slices(data, view=view, **options)
    figure = animation_to_plotly(animation, play=False)
    return PlotResult(figure, None, list(figure.data))
