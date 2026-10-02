"""LaTeX versions of the plots, as TikZ/pgfplots code: ``backend="tikz"``.

Every accessor plot that draws with Matplotlib (``array.plasma.plot.*``, ``dataset.plasma.plot.*``,
``out.plot.*``) takes ``backend="tikz"``. The plot is drawn with Matplotlib as usual, off screen,
and the drawn figure is converted by `maxplotlib <https://github.com/max-models/maxplotlib>`_
(:func:`maxplotlib.backends.tikzfigure.figure_to_tikz`) into a
`tikzfigure <https://github.com/max-models/tikzfigure>`_ figure of pgfplots axes: every panel at
the place and size it has in the Matplotlib figure, with its limits, log scales, labels and title
typeset by LaTeX, its ticks, legend and colorbar. Lines, markers, fits, reference lines, bars,
spans, contour lines and text are pgfplots code; meshes (also on mapped grids), images and the
color strips of colorbars are images that Matplotlib renders, placed with ``\\addplot graphics``.

>>> result = energy.plasma.plot.timeseries(fit=(0.0, 5.0), backend="tikz")
>>> # a standalone LaTeX document; also .tikz, .pdf, .png
>>> result.save("energy.tex")
>>> result.fig  # a tikzfigure.TikzFigure
>>> print(result.fig.generate_tikz())  # the tikzpicture, to paste into a paper
>>> plasma_plots.set_backend("tikz")  # the default for every plot from now on

Saving as ``.tikz`` writes the ``tikzpicture`` (for ``\\input`` into a document that loads
``pgfplots``), ``.tex`` a standalone document; both write the images the figure refers to next to
the file. ``.pdf`` and ``.png`` compile the figure with ``pdflatex``. Text is converted from
Matplotlib's: mathtext (``$\\omega/\\omega_{ci}$``) is LaTeX math, and plain text is escaped.

Animations and interactive viewers have no TikZ version. maxplotlib is optional:
``pip install "plasma-plots[tikz]"``.
"""

from __future__ import annotations

#: Options of the conversion, passed to :func:`maxplotlib.backends.tikzfigure.figure_to_tikz`:
#: the resolution of the parts drawn as images, and the sizes above which a scatter plot, a
#: collection or a line is drawn as an image instead of pgfplots code (TeX has limited memory).
#: Change them for every plot, e.g. ``plasma_plots.tikz_backend.OPTIONS["raster_dpi"] = 600``.
OPTIONS = {
    "raster_dpi": 300,
    "max_markers": 2000,
    "max_items": 500,
    "max_points": 20000,
}


def _maxplotlib():
    try:
        from maxplotlib.backends.tikzfigure import figure_to_tikz
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            'backend="tikz" needs maxplotlib: pip install "plasma-plots[tikz]" '
            "(or pip install maxplotlibx)"
        ) from error
    return figure_to_tikz


def to_tikz(figure, **options):
    """Convert a drawn Matplotlib figure into a TikZ/pgfplots figure.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure, e.g. ``result.fig`` of a plotting function.
    **options
        Passed to :func:`maxplotlib.backends.tikzfigure.figure_to_tikz`, over :data:`OPTIONS`:
        ``raster_dpi``, ``max_markers``, ``max_items``, ``max_points``, ``precision``.

    Returns
    -------
    tikzfigure.TikzFigure
        The figure: ``generate_tikz()`` for the ``tikzpicture``, ``savefig`` to write ``.tikz``,
        ``.tex``, ``.pdf`` or ``.png``.

    See Also
    --------
    plasma_plots.plotting.PlotResult.to_tikz : The same for a plot's result.

    Examples
    --------
    >>> figure = to_tikz(plot_lineout(phi.isel(t=-1, eta2=0, eta3=0)).fig)
    >>> figure.savefig("phi.tex")
    """
    figure_to_tikz = _maxplotlib()
    return figure_to_tikz(figure, **{**OPTIONS, **options})


def _is_tikz(figure) -> bool:
    """Whether ``figure`` is a tikzfigure figure."""
    return type(figure).__module__.startswith("tikzfigure")


def _drawn_as_tikz(draw):
    """Run ``draw`` off screen and convert what it returns."""
    from .mpi import SkippedPlot
    from .plotly_backend import _is_animation, _offscreen
    from .plotting import InteractiveSliceViewer, PlotResult

    _maxplotlib()  # fail before drawing if maxplotlib is missing
    with _offscreen():
        result = draw()
        if isinstance(result, SkippedPlot) or result is None:
            return result
        if isinstance(result, PlotResult):
            if _is_tikz(result.fig):
                return result
            return _tikz_result(to_tikz(result.fig), result)
        if isinstance(result, InteractiveSliceViewer) or _is_animation(result):
            raise TypeError(
                "backend='tikz' draws still figures; an animation or viewer needs "
                "backend='matplotlib' or backend='plotly'"
            )
    raise TypeError(f"cannot convert a {type(result).__name__} to TikZ")


def _tikz_result(figure, result):
    from .plotting import PlotResult

    return PlotResult(figure, None, [], list(result.fit_results), dict(result.data))
