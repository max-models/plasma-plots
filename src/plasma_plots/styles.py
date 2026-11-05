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
