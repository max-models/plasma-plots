"""Optional plots that need a whole run.

Plots and diagnostics of a single array live on the array, see
:class:`~struphy.post_processing.xarray_accessors.StruphyAccessor`:
``out.em_fields.phi_log.struphy.plot.slice(...)``, or from a value returned by
``out.evaluate("em_fields/phi_log")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import accessors  # noqa: F401  (registers array.struphy)

if TYPE_CHECKING:
    from struphy.post_processing.output import Output


class ProfilePlots:
    """Plots of one run's timing regions (``out.profile.results``), as
    ``out.struphy.plot.profile.<kind>(...)``.

    Thin pass-throughs to `scope-profiler <https://pypi.org/project/scope-profiler/>`_'s own
    plotting functions -- see their docstrings for the full set of keyword arguments (``ranks``,
    ``include``/``exclude``, ``backend``, ``filepath``, ...). Each returns whatever scope-profiler
    itself returns: a ``(fig, axes)`` pair for the default matplotlib backend, or a Plotly figure
    with ``backend="plotly"``.
    """

    def __init__(self, output: "Output"):
        self._output = output

    def gantt(self, *, return_fig: bool = True, verbose: bool = False, **kwargs):
        """A timeline of every recorded region, one row per rank."""
        from scope_profiler.plotting_scripts import plot_gantt

        return plot_gantt(
            self._output.profile.results,
            return_fig=return_fig,
            verbose=verbose,
            **kwargs,
        )

    def flame(self, *, return_fig: bool = True, verbose: bool = False, **kwargs):
        """A flame chart reconstructing the call stack from region timings."""
        from scope_profiler.plotting_scripts import plot_flame

        return plot_flame(
            self._output.profile.results,
            return_fig=return_fig,
            verbose=verbose,
            **kwargs,
        )

    def callgraph(self, *, return_fig: bool = True, verbose: bool = False, **kwargs):
        """The explicit call graph (which region calls which), without timings."""
        from scope_profiler.plotting_scripts import plot_callgraph

        return plot_callgraph(
            self._output.profile.results,
            return_fig=return_fig,
            verbose=verbose,
            **kwargs,
        )


class OutputPlots:
    """Plots of a whole run, constructed as ``OutputPlots(out)``.

    They return plotting-library objects with ``.show()``
    and ``.save(path)``, titled with the run's numerical parameters. Plots of one product are
    methods of that product, e.g. ``out.kinetic_ions.orbits.struphy.plot.trajectories()``.
    """

    def __init__(self, output: "Output"):
        self._output = output

    def scalars(
        self, names=None, *, relative_to: str | None = None, logy: bool = False
    ):
        """Overview of the scalar time series in one axes.

        Parameters
        ----------
        names:
            Scalars to show; all by default.
        relative_to:
            Show every scalar divided by this one.
        logy:
            Logarithmic value axis.
        """
        from .plotting import plot_scalars

        return plot_scalars(
            self._output.scalars,
            names=names,
            relative_to=relative_to,
            logy=logy,
            run_label=self._output.label,
        )

    def equilibrium(self, ax=None):
        """Radial profiles of this run's fluid equilibrium (``out.equil``, ``out.domain``)."""
        from .plotting import plot_equilibrium_profile

        return plot_equilibrium_profile(self._output.equil, self._output.domain, ax=ax)

    def equilibrium_3d(self, *, scalars: str = "p0", cmap="viridis"):
        """Create a PyVista equilibrium view; call ``.show()`` on the returned plotter."""
        from .plotting import show_equilibrium

        return show_equilibrium(
            self._output.equil, self._output.domain, scalars=scalars, cmap=cmap
        )

    def domain_3d(
        self, *, n1: int = 8, n2: int = 32, n3: int = 32, surface: bool = True
    ):
        """A PyVista wireframe of this run's mapping (``out.domain``); call ``.show()`` on it."""
        from .pyvista_plots import pyvista_domain

        return pyvista_domain(self._output.domain, n1=n1, n2=n2, n3=n3, surface=surface)

    @property
    def profile(self) -> "ProfilePlots":
        """Plots of this run's timing regions, e.g. ``out.struphy.plot.profile.gantt()``.

        Needs the optional ``scope-profiler`` extra (``pip install "struphy-plots[profiling]"``)
        and a run recorded with ``sim.run(profiling_activated=True)``.
        """
        return ProfilePlots(self._output)

    def __call__(self, *args, **kwargs):
        """The quick default plot: an overview of every scalar time series."""
        return self.scalars(*args, **kwargs)


class OutputAnalysis:
    """Spectral diagnostics of a run's products, as ``out.analysis.<kind>(product, ...)``.

    ``product`` is a product name (``"mhd/velocity"``, evaluated with ``out.evaluate``) or an
    already selected ``xarray.DataArray``. Select a component, slice or time interval before
    transforming when you do not need the whole field: the selected values are loaded into
    memory. The same diagnostics, and more, are available on any array as
    ``array.struphy.analysis.<kind>(...)``; see :mod:`struphy_plots.spectral`.
    """

    def __init__(self, output: "Output"):
        self._output = output

    def _array(self, product):
        return self._output.evaluate(product) if isinstance(product, str) else product

    def fft(
        self, product, *, dim: str, detrend: bool = False, window: str | None = None
    ):
        """Two-sided Fourier coefficients along ``dim``; see :func:`struphy_plots.spectral.fft`."""
        from .spectral import fft

        return fft(self._array(product), dim=dim, detrend=detrend, window=window)

    def time_fft(self, product, *, detrend: bool = False, window: str | None = None):
        """One-sided temporal coefficients and per-bin power; see
        :func:`struphy_plots.spectral.time_fft`."""
        from .spectral import time_fft

        return time_fft(self._array(product), detrend=detrend, window=window)

    def filter_time(
        self, product, *, dims=None, omega_min: float = 1e-8, pad_bins: int = 0
    ):
        """Reconstruct the dominant temporal band; see :func:`struphy_plots.spectral.filter_time`."""
        from .spectral import filter_time

        return filter_time(
            self._array(product), dims=dims, omega_min=omega_min, pad_bins=pad_bins
        )

    def mode_spectrum(
        self, product, *, dims=("e2", "e3"), names=("m", "n"), periods=1.0
    ):
        """Complex amplitudes over poloidal/toroidal mode numbers; see
        :func:`struphy_plots.spectral.mode_spectrum`."""
        from .spectral import mode_spectrum

        return mode_spectrum(
            self._array(product), dims=dims, names=names, periods=periods
        )


def _register_output_plot_property():
    """Wire ``out.plot`` to :class:`OutputPlots`, if struphy is installed.

    Guarded so importing ``struphy_plots`` stays optional: the array-level accessor works on any
    labeled ``xarray`` object without struphy installed at all.
    """
    try:
        from struphy.post_processing.output import Output
    except ImportError:
        return
    if not isinstance(Output.__dict__.get("plot"), property):
        Output.plot = property(OutputPlots)
    if not isinstance(Output.__dict__.get("analysis"), property):
        Output.analysis = property(OutputAnalysis)


_register_output_plot_property()
