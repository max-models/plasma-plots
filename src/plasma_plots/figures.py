"""Several plots in one figure, with either backend: ``plasma_plots.figure(...)``.

>>> with plasma_plots.figure(2, 1, sharex=True, backend="plotly") as fig:
...     energy.plasma.plot.timeseries(fit=(0.0, 5.0), ax=fig[0])
...     drift.plasma.plot.timeseries(logy=True, ax=fig[1])
>>> fig.save("energies.html")

Every plot method that takes ``ax=`` draws into one panel. The figure is drawn with Matplotlib and,
with ``backend="plotly"``, converted to one Plotly figure when the block ends, so the panels share
zoom where their axes are shared and every panel keeps its colorbar and legend.
"""

from __future__ import annotations

import numpy as np

from .plotly_backend import _collecting, _drawing, _offscreen, resolve_backend


class Figure:
    """A figure of several panels, made by :func:`figure`.

    Index it for a panel's Matplotlib axes (``fig[0]``, ``fig[1, 2]``) and pass that as ``ax=`` to
    any plot method. After the ``with`` block, it saves and shows like a
    :class:`~plasma_plots.plotting.PlotResult`.

    Parameters
    ----------
    nrows, ncols : int
        The number of rows and columns of panels.
    backend : {"matplotlib", "plotly"} or None
        How the figure is finished; ``None`` for the default (see :func:`plasma_plots.set_backend`).
    sharex, sharey : bool or {"row", "col", "all"}
        Share the horizontal or vertical axes.
    figsize : (float, float) or None
        The size in inches; ``None`` for one from the number of panels.
    title : str or None
        A title above all panels.
    **options
        Passed on to ``matplotlib.pyplot.subplots``.

    Attributes
    ----------
    axes : numpy.ndarray of matplotlib.axes.Axes or None
        The panels, ``(nrows, ncols)``; ``None`` on MPI ranks other than 0, where nothing is drawn.
    results : list of PlotResult
        What the plot methods drawn into the panels returned, in order (e.g. their
        ``fit_results``).
    """

    def __init__(
        self, nrows, ncols, *, backend, sharex, sharey, figsize, title, **options
    ):
        from .mpi import is_plotting_rank

        self.backend = backend
        self._plotly = (
            resolve_backend(backend) == "plotly"
        )  # before the block makes every plot Matplotlib
        self.results = []
        self._result = None
        self._title = title
        self.axes = None
        self._token = None
        if not is_plotting_rank():
            return
        import matplotlib.pyplot as plt

        from .plotting import PLOT_STYLE

        if figsize is None:
            figsize = (
                PLOT_STYLE["figure.figsize"][0] * min(ncols, 2) * 0.75
                + 2.0 * (ncols > 2) * (ncols - 2),
                3.2 * nrows + 0.8,
            )
        with plt.rc_context(PLOT_STYLE):
            self._fig, axes = plt.subplots(
                nrows,
                ncols,
                sharex=sharex,
                sharey=sharey,
                figsize=figsize,
                squeeze=False,
                layout="constrained",
                **options,
            )
        self._fig._plasma_composed = (
            True  # the plots in it leave the title and layout to the figure
        )
        self.axes = axes

    def __getitem__(self, index):
        if self.axes is None:  # another MPI rank: the plots are skipped
            return None
        if isinstance(index, (int, np.integer)):
            return self.axes.ravel()[index]
        return self.axes[index]

    def __len__(self):
        return 0 if self.axes is None else self.axes.size

    def __enter__(self):
        # inside the block every plot draws with Matplotlib into its panel; the block converts once
        self._token = _drawing.set(True)
        _collecting.append(self.results)
        return self

    def __exit__(self, *exc):
        _collecting.remove(self.results)
        _drawing.reset(self._token)
        if exc[0] is None:
            self._finish()
        return False

    def _finish(self):
        self._result = self._drawn(final=True)

    def _drawn(self, *, final):
        """The figure as drawn so far; ``final`` hides empty panels and releases a converted figure."""
        from .mpi import SkippedPlot, mpi_rank
        from .plotting import PlotResult

        if self.axes is None:
            return SkippedPlot("figure", mpi_rank())
        fig = self._fig
        if self._title:
            fig.suptitle(self._title)
        empty = [
            ax
            for ax in self.axes.ravel()
            if not ax.has_data() and not ax.get_legend() and ax.get_visible()
        ]
        for ax in empty:  # panels left empty are not drawn
            ax.set_visible(False)
        fits = [
            fit for result in self.results for fit in getattr(result, "fit_results", [])
        ]
        try:
            if not self._plotly:
                return PlotResult(fig, self.axes, [], fits)
            import matplotlib.pyplot as plt

            from .plotly_backend import _plotly, to_plotly

            _plotly()
            with _offscreen():
                converted = to_plotly(fig)
            if final:
                plt.close(fig)
            return PlotResult(converted, None, list(converted.data), fits)
        finally:
            if (
                not final
            ):  # a snapshot inside the block: later plots may still fill these panels
                for ax in empty:
                    ax.set_visible(True)

    @property
    def result(self):
        """The figure, as a :class:`~plasma_plots.plotting.PlotResult`.

        After the ``with`` block the finished figure; inside it, the figure as drawn so far, so
        that ``fig.save(...)`` works in either place.

        Returns
        -------
        PlotResult or SkippedPlot
            The figure, Matplotlib or Plotly, with the ``fit_results`` of every panel.
        """
        if self._result is None:
            return self._drawn(final=False)
        return self._result

    @property
    def fig(self):
        """The finished Matplotlib or Plotly figure."""
        return self.result.fig

    def save(self, path, **kwargs):
        """Save the figure, as :meth:`PlotResult.save <plasma_plots.plotting.PlotResult.save>` does.

        Inside the ``with`` block it saves the panels drawn so far.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write; its extension picks the format.
        **kwargs
            Passed to :meth:`PlotResult.save <plasma_plots.plotting.PlotResult.save>`.

        Returns
        -------
        str
            The path written.
        """
        return self.result.save(path, **kwargs)

    def show(self):
        """Show the finished figure.

        Returns
        -------
        Figure
            This figure.
        """
        self.result.show()
        return self

    def _ipython_display_(self):
        self.result._ipython_display_()

    def __repr__(self):
        shape = None if self.axes is None else self.axes.shape
        return f"{type(self).__name__}(panels={shape}, backend={self.backend!r})"


def figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    backend: str | None = None,
    sharex: bool | str = False,
    sharey: bool | str = False,
    figsize=None,
    title: str | None = None,
    **options,
) -> Figure:
    """Compose several plots into one figure, drawn with Matplotlib or as one Plotly figure.

    Use it as a ``with`` block: every plot method given ``ax=fig[i]`` draws into panel ``i``;
    at the end of the block the figure is finished, and with ``backend="plotly"`` converted once.

    Parameters
    ----------
    nrows : int, optional
        The number of rows of panels. Default: 1.
    ncols : int, optional
        The number of columns of panels. Default: 1.
    backend : {"matplotlib", "plotly"}, optional
        Finish the figure as a Matplotlib or as an interactive Plotly figure. Default: the one set
        with :func:`plasma_plots.set_backend`, ``"matplotlib"`` unless changed.
    sharex : bool or {"row", "col", "all"}, optional
        Share the horizontal axes (and their zoom, in Plotly), as for ``matplotlib.pyplot.subplots``.
        Default: ``False``.
    sharey : bool or {"row", "col", "all"}, optional
        Share the vertical axes. Default: ``False``.
    figsize : (float, float), optional
        The size in inches. Default: from the number of panels.
    title : str, optional
        A title above all panels. Default: none.
    **options
        Passed on to ``matplotlib.pyplot.subplots``, e.g. ``gridspec_kw={"height_ratios": [3, 1]}``.

    Returns
    -------
    Figure
        The figure; index it for the panels' axes, and save or show it after the block.

    Examples
    --------
    >>> with plasma_plots.figure(2, 1, sharex=True, backend="plotly") as fig:
    ...     energy.plasma.plot.timeseries(fit=(0.0, 5.0), ax=fig[0])
    ...     drift.plasma.plot.timeseries(logy=True, ax=fig[1])
    >>> fig.save("energies.html")
    >>> fig.results[0].fit_results[0].rate
    """
    return Figure(
        nrows,
        ncols,
        backend=backend,
        sharex=sharex,
        sharey=sharey,
        figsize=figsize,
        title=title,
        **options,
    )
