"""Animated SVG header pictures for the home pages of the docs: ``plasma_plots.hero``.

The max-models docs theme (starlight-theme) shows ``docs/src/assets/hero.svg``, when a project
has one, across the top of its home page instead of the default streamlines. The picture takes the
site's colours: every mark has a tone (``"accent"``, ``"high"``, ``"low"``, ``"muted"`` or
``"bg"``) that the theme turns into a colour of the palette, in light and dark mode. Moving parts
are SVG animations, which the theme stops for readers who prefer reduced motion.

>>> import numpy as np
>>> from plasma_plots.hero import HeroFigure
>>> fig = HeroFigure(xlim=(0, 4), ylim=(-1, 1))
>>> x = np.linspace(0, 4, 200)
>>> line = fig.line(x, np.sin(np.pi * x), opacity=0.8)
>>> dot = fig.tracer(line, duration=6)  # a dot drifting along the line
>>> fig.save("docs/src/assets/hero.svg")  # doctest: +SKIP

Coordinates are data coordinates within ``xlim`` and ``ylim`` (``y`` pointing up); radii and line
widths are in pixels of the ``width`` x ``height`` picture. The theme crops the picture to the
header (``preserveAspectRatio="xMidYMid slice"``) and fades its left part behind the title, so the
interesting part goes to the right.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

TONES = ("accent", "high", "low", "muted", "bg")


def _check_tone(tone):
    if tone not in TONES:
        raise ValueError(f"tone must be one of {TONES}, not {tone!r}")
    return tone


def _fmt(value):
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


class Mark:
    """One element of a :class:`HeroFigure`, returned by its drawing methods.

    Parameters
    ----------
    tag : str
        The SVG element.
    attrs : dict
        Its attributes.
    path : str or None
        The SVG path of a line, which :meth:`HeroFigure.tracer` follows.
    """

    def __init__(self, tag, attrs, path=None):
        self.tag = tag
        self.attrs = attrs
        self.path = path
        self.children = []

    def animate(self, attribute, values, duration, begin=0.0):
        """Animate an attribute through a list of values, on a loop.

        Parameters
        ----------
        attribute : str
            The SVG attribute, e.g. ``"opacity"``, ``"r"`` or ``"cx"``.
        values : sequence
            The values it passes through, evenly spaced in time; repeat the first at the end for a
            smooth loop.
        duration : float
            Seconds for one pass.
        begin : float
            Seconds into the pass at which the page starts (spreads several marks apart).

        Returns
        -------
        Mark
            This mark, to chain calls.
        """
        text = ";".join(
            _fmt(v) if isinstance(v, (int, float, np.number)) else str(v)
            for v in values
        )
        self.children.append(
            f'<animate attributeName="{attribute}" values="{text}" dur="{_fmt(duration)}s" '
            f'begin="-{_fmt(begin)}s" repeatCount="indefinite"/>'
        )
        return self

    def to_svg(self):
        """Return the element as SVG text.

        Returns
        -------
        str
            The element.
        """
        attrs = " ".join(
            f'{key}="{escape(str(value))}"' for key, value in self.attrs.items()
        )
        if not self.children:
            return f"<{self.tag} {attrs}/>"
        return f"<{self.tag} {attrs}>{''.join(self.children)}</{self.tag}>"


class HeroFigure:
    """An animated SVG picture for the header of a docs home page.

    Parameters
    ----------
    xlim, ylim : (float, float)
        The data range shown across and up the picture.
    width, height : int
        The size of the picture in pixels (the theme scales it to the header).
    """

    def __init__(self, xlim=(0.0, 1.0), ylim=(0.0, 1.0), width=1600, height=600):
        self.xlim = tuple(float(v) for v in xlim)
        self.ylim = tuple(float(v) for v in ylim)
        self.width = int(width)
        self.height = int(height)
        self.marks = []

    def _xy(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        px = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.width
        py = (self.ylim[1] - y) / (self.ylim[1] - self.ylim[0]) * self.height
        return px, py

    def _scale(self, length):
        return abs(length) / (self.xlim[1] - self.xlim[0]) * self.width

    def _add(self, mark, title):
        if title is not None:
            mark.children.append(f"<title>{escape(title)}</title>")
        self.marks.append(mark)
        return mark

    def _path(self, x, y, closed=False):
        px, py = self._xy(x, y)
        keep = np.isfinite(px) & np.isfinite(py)
        points = [f"{_fmt(a)} {_fmt(b)}" for a, b in zip(px[keep], py[keep])]
        if not points:
            raise ValueError("a line needs at least one finite point")
        return "M" + " L".join(points) + (" Z" if closed else "")

    def line(self, x, y, tone="accent", width=1.5, opacity=1.0, dash=None, title=None):
        """Draw a line through points.

        Parameters
        ----------
        x, y : array_like
            The points, in data coordinates; non-finite points are left out.
        tone : str
            Its colour (see the module docstring).
        width : float
            The line width in pixels.
        opacity : float
            Between 0 and 1.
        dash : str or None
            An SVG dash pattern, e.g. ``"4 6"``.
        title : str or None
            A tooltip shown on hover.

        Returns
        -------
        Mark
            The line, to animate or to pass to :meth:`tracer`.
        """
        d = self._path(x, y)
        attrs = {
            "d": d,
            "class": f"s-{_check_tone(tone)}",
            "fill": "none",
            "stroke-width": _fmt(width),
        }
        if opacity != 1:
            attrs["stroke-opacity"] = f"{opacity:.2f}"
        if dash:
            attrs["stroke-dasharray"] = dash
        return self._add(Mark("path", attrs, path=d), title)

    def polygon(self, x, y, tone="accent", opacity=1.0, outline=None, title=None):
        """Fill a closed shape.

        Parameters
        ----------
        x, y : array_like
            The corners, in data coordinates.
        tone : str
            The fill colour.
        opacity : float
            The fill opacity.
        outline : str or None
            The tone of an outline, or ``None`` for none.
        title : str or None
            A tooltip shown on hover.

        Returns
        -------
        Mark
            The shape.
        """
        classes = f"f-{_check_tone(tone)}" + (
            f" s-{_check_tone(outline)}" if outline else ""
        )
        attrs = {"d": self._path(x, y, closed=True), "class": classes}
        if opacity != 1:
            attrs["fill-opacity"] = f"{opacity:.2f}"
        return self._add(Mark("path", attrs), title)

    def rect(
        self, x0, y0, x1, y1, tone="accent", opacity=1.0, outline=None, title=None
    ):
        """Draw a rectangle between two corners.

        Parameters
        ----------
        x0, y0, x1, y1 : float
            Two opposite corners, in data coordinates.
        tone : str
            The fill colour.
        opacity : float
            The fill opacity.
        outline : str or None
            The tone of an outline, or ``None`` for none.
        title : str or None
            A tooltip shown on hover.

        Returns
        -------
        Mark
            The rectangle.
        """
        return self.polygon(
            [x0, x1, x1, x0], [y0, y0, y1, y1], tone, opacity, outline, title
        )

    def circle(
        self, x, y, r, tone="accent", fill=True, opacity=1.0, width=1.5, title=None
    ):
        """Draw a circle.

        Parameters
        ----------
        x, y : float
            The centre, in data coordinates.
        r : float
            The radius in pixels.
        tone : str
            Its colour.
        fill : bool
            Filled, or only the outline.
        opacity : float
            Between 0 and 1.
        width : float
            The outline width in pixels, when not filled.
        title : str or None
            A tooltip shown on hover.

        Returns
        -------
        Mark
            The circle.
        """
        px, py = self._xy(x, y)
        tone = _check_tone(tone)
        attrs = {"cx": _fmt(px), "cy": _fmt(py), "r": _fmt(r)}
        if fill:
            attrs["class"] = f"f-{tone}"
        else:
            attrs.update(
                {"class": f"s-{tone}", "fill": "none", "stroke-width": _fmt(width)}
            )
        if opacity != 1:
            attrs["opacity"] = f"{opacity:.2f}"
        return self._add(Mark("circle", attrs), title)

    def disc(self, x, y, radius, tone="bg", outline="accent", width=2.5, title=None):
        """Draw a solid body, such as an obstacle or an electrode, with a radius in data units.

        Parameters
        ----------
        x, y : float
            The centre, in data coordinates.
        radius : float
            The radius in data units (of ``x``).
        tone : str
            The fill colour.
        outline : str
            The outline colour.
        width : float
            The outline width in pixels.
        title : str or None
            A tooltip shown on hover.

        Returns
        -------
        Mark
            The disc.
        """
        px, py = self._xy(x, y)
        classes = f"f-{_check_tone(tone)} s-{_check_tone(outline)}"
        attrs = {
            "cx": _fmt(px),
            "cy": _fmt(py),
            "r": _fmt(self._scale(radius)),
            "class": classes,
        }
        attrs["stroke-width"] = _fmt(width)
        return self._add(Mark("circle", attrs), title)

    def tracer(self, along, duration, offset=0.0, r=4.0, tone="high", y=None):
        """Add a dot that moves along a line or a trajectory, on a loop.

        Parameters
        ----------
        along : Mark or array_like
            A line from :meth:`line`, or the ``x`` of a trajectory (with ``y``).
        duration : float
            Seconds for one pass.
        offset : float
            Seconds into the pass at which the page starts.
        r : float
            The radius in pixels.
        tone : str
            Its colour.
        y : array_like or None
            The ``y`` of the trajectory, when ``along`` is its ``x``.

        Returns
        -------
        Mark
            The dot, e.g. to animate its colour or size too.
        """
        path = along.path if isinstance(along, Mark) else self._path(along, y)
        if path is None:
            raise ValueError(
                "a tracer follows a line, or a trajectory given as x and y"
            )
        dot = Mark("circle", {"r": _fmt(r), "class": f"f-{_check_tone(tone)}"})
        dot.children.append(
            f'<animateMotion dur="{_fmt(duration)}s" begin="-{_fmt(offset)}s" '
            f'repeatCount="indefinite" path="{path}"/>'
        )
        self.marks.append(dot)
        return dot

    def to_svg(self):
        """Return the picture as SVG text.

        Returns
        -------
        str
            The SVG document.
        """
        body = "\n".join(mark.to_svg() for mark in self.marks)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'preserveAspectRatio="xMidYMid slice">\n{body}\n</svg>\n'
        )

    def save(self, path):
        """Write the picture to an SVG file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file, usually ``docs/src/assets/hero.svg``.

        Returns
        -------
        pathlib.Path
            The file written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_svg())
        return path
