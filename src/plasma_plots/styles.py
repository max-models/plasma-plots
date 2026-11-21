"""Named plot styles and a colour-blind-safe palette: ``plasma_plots.set_style``.

Every Matplotlib plot of plasma-plots draws with the current style. ``"default"`` is the style the
plots always had; the others change it:

- ``"colorblind"``: the default with the Okabe–Ito palette (:data:`PALETTE`) as the line colours,
  which stay distinct for the common forms of colour blindness.
- ``"paper"``: small figures with small text for a journal column, in the same palette.
- ``"dark"``: light lines and text on a dark background, for slides and dark docs pages.
- ``"bare"``: no axes, ticks, labels or titles, a transparent background and the accent colour
  first, for header pictures and front pages (see :func:`plasma_plots.hero.from_array` for an
  animated SVG and :func:`plasma_plots.plotly_backend.bare` for a Plotly figure).

:func:`set_style` sets the style of every later plot; :func:`style` sets it for a ``with`` block
only, where it also applies to plain Matplotlib figures created inside the block.

Examples
--------
>>> import plasma_plots
>>> plasma_plots.set_style("colorblind")
>>> with plasma_plots.style("dark"):
...     phi.plasma.plot.slice(t=-1, eta3=0).save("phi_dark.png")
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar

from cycler import cycler

PALETTE = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)
"""The Okabe–Ito palette: eight colours that colour-blind readers tell apart."""

ACCENT = "#2a78d6"
"""The accent colour of the max-models docs, the first line colour of the ``"bare"`` style."""

_DARK_BACKGROUND = "#17181c"
_DARK_TEXT = "#e6e6e6"

DEFAULT = {
    "figure.figsize": (8.0, 5.0),
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.titlesize": "medium",
    "legend.frameon": False,
    "image.cmap": "viridis",
}

_COLORBLIND = {**DEFAULT, "axes.prop_cycle": cycler(color=PALETTE)}

STYLES = {
    "default": DEFAULT,
    "colorblind": _COLORBLIND,
    "paper": {
        **_COLORBLIND,
        "figure.figsize": (3.4, 2.4),
        "figure.dpi": 200,
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "savefig.dpi": 300,
    },
    "dark": {
        **DEFAULT,
        "axes.prop_cycle": cycler(color=(*PALETTE[:7], _DARK_TEXT)),
        "figure.facecolor": _DARK_BACKGROUND,
        "axes.facecolor": _DARK_BACKGROUND,
        "savefig.facecolor": _DARK_BACKGROUND,
        "axes.edgecolor": "#8a8d96",
        "axes.labelcolor": _DARK_TEXT,
        "axes.titlecolor": _DARK_TEXT,
        "text.color": _DARK_TEXT,
        "xtick.color": _DARK_TEXT,
        "ytick.color": _DARK_TEXT,
        "grid.color": "#ffffff",
        "grid.alpha": 0.15,
        "legend.labelcolor": _DARK_TEXT,
    },
    "bare": {
        **DEFAULT,
        "axes.prop_cycle": cycler(color=(ACCENT, *PALETTE[1:])),
        "axes.grid": False,
        "axes.spines.left": False,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.spines.bottom": False,
        "xtick.bottom": False,
        "xtick.labelbottom": False,
        "ytick.left": False,
        "ytick.labelleft": False,
        "axes.labelcolor": "none",
        "axes.titlecolor": "none",
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.transparent": True,
        "lines.linewidth": 2.0,
        "legend.frameon": False,
    },
}
"""The named styles: Matplotlib ``rcParams`` per name."""

# the style of set_style, a module global like the backend of set_backend, so that it lasts
# across notebook cells; a `with style(...)` block sets its own for the code inside it
_default = ("default", None)
_block = ContextVar("plasma_plots_style", default=None)


def _active():
    """``(name, overrides)`` of the current style."""
    return _block.get() or _default


def _check(name):
    if name not in STYLES:
        raise ValueError(f"unknown style {name!r}; expected one of {sorted(STYLES)}")
    return name


def get_style() -> str:
    """Return the name of the current style.

    Returns
    -------
    str
        One of the keys of :data:`STYLES`.
    """
    return _active()[0]


def style_params(name: str | None = None) -> dict:
    """Return the Matplotlib ``rcParams`` of a style.

    Parameters
    ----------
    name : str, optional
        The style. Default: the current one, with the overrides given to :func:`set_style` or
        :func:`style`.

    Returns
    -------
    dict
        The parameters, to pass to ``matplotlib.pyplot.rc_context``.

    Raises
    ------
    ValueError
        If there is no style of that name.
    """
    if name is not None:
        return dict(STYLES[_check(name)])
    current, overrides = _active()
    return {**STYLES[current], **(overrides or {})}


def _rc_names(overrides):
    """Matplotlib's names for keyword overrides, e.g. ``axes.spines.left`` for ``axes_spines_left``."""
    if not overrides:
        return None
    import matplotlib

    names = {key.replace(".", "_"): key for key in matplotlib.rcParams}
    unknown = sorted(set(overrides) - set(names))
    if unknown:
        raise ValueError(f"unknown Matplotlib rcParams {unknown}")
    return {names[key]: value for key, value in overrides.items()}


def set_style(name: str = "default", **overrides) -> None:
    """Set the style of every later plot.

    Parameters
    ----------
    name : {"default", "colorblind", "paper", "dark", "bare"}, optional
        The style (see :data:`STYLES`). Default: ``"default"``, the plots' usual style.
    **overrides
        Matplotlib ``rcParams`` to change on top of the style, with dots written as underscores,
        e.g. ``figure_dpi=150``.

    Raises
    ------
    ValueError
        If there is no style of that name, or an override is not a Matplotlib ``rcParams`` name.

    See Also
    --------
    style : The same for a ``with`` block only.

    Examples
    --------
    >>> plasma_plots.set_style("colorblind")
    >>> plasma_plots.set_style("paper", figure_figsize=(3.4, 3.0))
    """
    global _default
    _default = (_check(name), _rc_names(overrides))


@contextmanager
def style(name: str, **overrides):
    """Draw with a style inside a ``with`` block.

    Inside the block the style also applies to plain Matplotlib figures, e.g. a colorbar or an
    annotation added to a plot of plasma-plots.

    Parameters
    ----------
    name : {"default", "colorblind", "paper", "dark", "bare"}
        The style (see :data:`STYLES`).
    **overrides
        Matplotlib ``rcParams`` to change on top of the style, with dots written as underscores,
        e.g. ``figure_dpi=150``.

    Yields
    ------
    dict
        The ``rcParams`` in effect.

    Raises
    ------
    ValueError
        If there is no style of that name, or an override is not a Matplotlib ``rcParams`` name.

    Examples
    --------
    >>> with plasma_plots.style("dark"):
    ...     energy.plasma.plot.timeseries().save("energy_dark.png")
    """
    import matplotlib
    import matplotlib.pyplot as plt

    token = _block.set((_check(name), _rc_names(overrides)))
    interactive = None
    try:
        params = style_params()
        with plt.rc_context(params):
            try:
                yield params
            finally:
                interactive = matplotlib.is_interactive()
    finally:
        _block.reset(token)
        # IPython may switch interactive mode on inside the block; keep it (see _plot_style)
        if interactive is not None:
            matplotlib.interactive(interactive)


def _luminance(color) -> float | None:
    """The relative luminance of a colour, or None for no colour (``"none"``, fully transparent)."""
    from matplotlib.colors import to_rgba

    try:
        r, g, b, a = to_rgba(color)
    except (ValueError, TypeError):
        return None
    if a == 0:
        return None
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _swap(color, ink, paper):
    """Black and dark grey as ``ink``, white as ``paper``, keeping the alpha.

    Dark colours with a hue (e.g. the dark end of viridis) stay: they belong to a colour scale.
    """
    from matplotlib.colors import to_rgba

    lum = _luminance(color)
    if lum is None:
        return color
    r, g, b, alpha = to_rgba(color)
    if lum < 0.2 and max(r, g, b) - min(r, g, b) < 0.1:
        return to_rgba(ink, alpha)
    if lum > 0.97:
        return to_rgba(paper, alpha)
    return color


def to_dark(
    fig,
    *,
    text: str = _DARK_TEXT,
    axes: str = "#8a8d96",
    background: str = _DARK_BACKGROUND,
):
    """Recolour a drawn Matplotlib figure for a dark page, in place.

    For a dark variant of a figure that was drawn for a white page (e.g. ``x.dark.png`` next to
    ``x.png``), without drawing it again: black and dark grey text, lines, markers, edges and
    hatches become ``text``, the spines and ticks ``axes``, white becomes ``background``, and the
    figure and axes backgrounds become transparent. Colormapped artists (images, meshes,
    scatters coloured by a value) and other colours stay as they are. Save it with
    ``transparent=True`` to show the page behind it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure; it is changed.
    text : str, optional
        The colour of text and of black lines. Default: the ``"dark"`` style's text colour.
    axes : str, optional
        The colour of spines and ticks. Default: a mid grey.
    background : str, optional
        The colour for white (e.g. white markers or masks). Default: the ``"dark"`` style's
        background.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure.

    Examples
    --------
    >>> result = energy.plasma.plot.timeseries()
    >>> result.fig.savefig("energy.png")
    >>> to_dark(result.fig).savefig("energy.dark.png", transparent=True)
    """
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    fig.patch.set_facecolor("none")
    for ax in fig.axes:
        ax.patch.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_edgecolor(axes)
        ax.tick_params(which="both", colors=axes, labelcolor=text)
    backgrounds = {fig.patch, *(ax.patch for ax in fig.axes)}
    for artist in fig.findobj():
        if isinstance(artist, Text):
            artist.set_color(_swap(artist.get_color(), text, background))
            box = artist.get_bbox_patch()
            if box is not None:
                box.set_facecolor(_swap(box.get_facecolor(), text, background))
                box.set_edgecolor(_swap(box.get_edgecolor(), axes, background))
        elif isinstance(artist, Line2D):
            artist.set_color(_swap(artist.get_color(), text, background))
            artist.set_markerfacecolor(
                _swap(artist.get_markerfacecolor(), text, background)
            )
            artist.set_markeredgecolor(
                _swap(artist.get_markeredgecolor(), text, background)
            )
        elif isinstance(artist, Patch) and artist not in backgrounds:
            # legend frames and boxes: white as the page; black outlines as the axes
            artist.set_facecolor(_swap(artist.get_facecolor(), text, background))
            artist.set_edgecolor(_swap(artist.get_edgecolor(), axes, background))
        elif isinstance(artist, Collection) and artist.get_array() is None:
            # single-coloured collections (contour lines, scatters, quivers); colormapped ones stay
            faces, edges = artist.get_facecolors(), artist.get_edgecolors()
            if len(faces):
                artist.set_facecolors([_swap(c, text, background) for c in faces])
            if len(edges):
                artist.set_edgecolors([_swap(c, text, background) for c in edges])
    return fig
