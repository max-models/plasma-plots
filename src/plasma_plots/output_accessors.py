"""Plots and diagnostics of a whole run, as ``out.plot`` and ``out.analysis``.

Importing ``plasma_plots`` adds two properties to struphy's ``Output`` (when struphy is
installed; without importing struphy itself, which takes seconds: the properties are attached
when struphy's output module is imported, before or after ``plasma_plots``): ``out.plot`` (:class:`OutputPlots`) for optional plots that need a whole run, and
``out.analysis`` (:class:`OutputAnalysis`) for spectral diagnostics of its products.

Plots and diagnostics of a single array live on the array, see
:class:`~plasma_plots.accessors.PlasmaAccessor`:
``out.em_fields.phi_log.plasma.plot.slice(...)``, or from a value returned by
``out.evaluate("em_fields/phi_log")``.
"""

from __future__ import annotations

import importlib.abc
import sys
from typing import TYPE_CHECKING

import numpy as np

from . import accessors  # noqa: F401  (registers array.plasma)
from .accessors import Backend
from .mpi import rank_zero
from .plotly_backend import with_backend

if TYPE_CHECKING:
    from struphy.post_processing.output import Output


class ProfilePlots:
    """Plots of one run's timing regions (``out.profile.results``), as ``out.plot.profile.<kind>(...)``.

    Thin pass-throughs to `scope-profiler <https://pypi.org/project/scope-profiler/>`_'s own
    plotting functions -- see their docstrings for the full set of keyword arguments (``ranks``,
    ``include``/``exclude``, ``backend``, ``filepath``, ...). Each returns whatever scope-profiler
    itself returns: a ``(fig, axes)`` pair for the default matplotlib backend, or a Plotly figure
    with ``backend="plotly"``.

    Parameters
    ----------
    output : Output
        The struphy run whose ``out.profile.results`` are plotted. Usually reached as
        ``out.plot.profile`` instead of constructed directly.

    Examples
    --------
    >>> out.plot.profile.gantt()
    >>> out.plot.profile.flame(backend="plotly")
    """

    def __init__(self, output: "Output"):
        self._output = output

    @rank_zero
    def gantt(self, *, return_fig: bool = True, verbose: bool = False, **kwargs):
        """A timeline of every recorded region, one row per rank.

        Parameters
        ----------
        return_fig : bool, optional
            Return the rendered figure (scope-profiler's own default is to return ``None``).
            Default: ``True``.
        verbose : bool, optional
            Let scope-profiler print progress information. Default: ``False``.
        **kwargs
            Passed on to scope-profiler's ``plot_gantt``
            (``ranks``, ``include``/``exclude``, ``backend``, ``filepath``, ``min_duration``, ...).

        Returns
        -------
        tuple or plotly.graph_objects.Figure or None
            Whatever scope-profiler returns: ``(fig, axes)`` for the default matplotlib backend,
            a Plotly figure with ``backend="plotly"``, ``None`` with ``return_fig=False``.

        Examples
        --------
        >>> out.plot.profile.gantt()
        """
        from scope_profiler.plotting_scripts import plot_gantt

        return plot_gantt(
            self._output.profile.results,
            return_fig=return_fig,
            verbose=verbose,
            **kwargs,
        )

    @rank_zero
    def flame(self, *, return_fig: bool = True, verbose: bool = False, **kwargs):
        """A flame chart reconstructing the call stack from region timings.

        Parameters
        ----------
        return_fig : bool, optional
            Return the rendered figure (scope-profiler's own default is to return ``None``).
            Default: ``True``.
        verbose : bool, optional
            Let scope-profiler print progress information. Default: ``False``.
        **kwargs
            Passed on to scope-profiler's ``plot_flame``
            (``ranks``, ``include``/``exclude``, ``backend``, ``filepath``, ...).

        Returns
        -------
        tuple or plotly.graph_objects.Figure or None
            Whatever scope-profiler returns: ``(fig, axes)`` for the default matplotlib backend,
            a Plotly figure with ``backend="plotly"``, ``None`` with ``return_fig=False``.

        Examples
        --------
        >>> out.plot.profile.flame()
        """
        from scope_profiler.plotting_scripts import plot_flame

        return plot_flame(
            self._output.profile.results,
            return_fig=return_fig,
            verbose=verbose,
            **kwargs,
        )

    @rank_zero
    def callgraph(self, *, return_fig: bool = True, verbose: bool = False, **kwargs):
        """The explicit call graph (which region calls which), without timings.

        Parameters
        ----------
        return_fig : bool, optional
            Return the rendered figure (scope-profiler's own default is to return ``None``).
            Default: ``True``.
        verbose : bool, optional
            Let scope-profiler print progress information. Default: ``False``.
        **kwargs
            Passed on to scope-profiler's ``plot_callgraph``
            (``rank``, ``include``/``exclude``, ``backend``, ``compact``, ``fluid``, ...).

        Returns
        -------
        tuple or plotly.graph_objects.Figure or None
            Whatever scope-profiler returns: ``(fig, axes)`` for the default matplotlib backend,
            a Plotly figure with ``backend="plotly"``, ``None`` with ``return_fig=False``.

        Examples
        --------
        >>> out.plot.profile.callgraph()
        """
        from scope_profiler.plotting_scripts import plot_callgraph

        return plot_callgraph(
            self._output.profile.results,
            return_fig=return_fig,
            verbose=verbose,
            **kwargs,
        )


class OutputPlots:
    """Plots of a whole run, as ``out.plot.<kind>(...)``, constructed as ``OutputPlots(out)``.

    They return plotting-library objects with ``.show()``
    and ``.save(path)``, titled with the run's numerical parameters. Plots of one product are
    methods of that product, e.g. ``out.kinetic_ions.orbits.plasma.plot.trajectories()``.
    Calling ``out.plot()`` itself gives the quick default plot, :meth:`scalars`.

    Parameters
    ----------
    output : Output
        The struphy run to plot.

    Examples
    --------
    >>> out.plot()
    >>> out.plot.energies(logy=True)
    >>> out.plot.domain_3d().show()
    """

    def __init__(self, output: "Output"):
        self._output = output

    @with_backend
    def scalars(
        self,
        names=None,
        *,
        relative_to: str | None = None,
        logy: bool = False,
        backend: Backend | None = None,
    ):
        """Overview of the scalar time series in one axes.

        Parameters
        ----------
        names : list of str, optional
            Scalars to show; all by default.
        relative_to : str, optional
            Show every scalar divided by this one.
        logy : bool, optional
            Logarithmic value axis.
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`plasma_plots.plotly_backend`). Default: the one set with
            :func:`plasma_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn lines, titled with the run's label.

        See Also
        --------
        plasma_plots.plotting.plot_scalars : The function behind this method.

        Examples
        --------
        >>> out.plot.scalars()
        >>> out.plot.scalars(["en_U", "en_B"], logy=True)
        """
        from .plotting import plot_scalars

        return plot_scalars(
            self._output.scalars,
            names=names,
            relative_to=relative_to,
            logy=logy,
            run_label=self._output.label,
        )

    @with_backend
    def energies(
        self,
        *,
        parts=None,
        total: str | None = "en_tot",
        groups: dict | None = None,
        logy: bool = False,
        backend: Backend | None = None,
    ):
        """Plot the run's energy budget from its ``en_*`` scalars.

        Shows the ``en_*`` scalars, the relative drift of ``total``, and, with ``groups``
        (e.g. ``{"wave": ["en_U", "en_B", "en_p"], "energetic ions": ["en_fv", "en_fB"]}``), the
        energy exchanged between them.

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`plasma_plots.plotly_backend`). Default: the one set with
            :func:`plasma_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn lines, titled with the run's label.

        See Also
        --------
        plasma_plots.plotting.plot_energy_budget : The function behind this method.

        Examples
        --------
        >>> out.plot.energies()
        >>> out.plot.energies(
        ...     groups={
        ...         "wave": ["en_U", "en_B", "en_p"],
        ...         "energetic ions": ["en_fv", "en_fB"],
        ...     }
        ... )
        """
        from .plotting import plot_energy_budget

        return plot_energy_budget(
            self._output.scalars,
            parts=parts,
            total=total,
            groups=groups,
            logy=logy,
            run_label=self._output.label,
        )

    @with_backend
    def equilibrium(self, ax=None, *, backend: Backend | None = None):
        """Plot radial profiles of this run's fluid equilibrium (``out.equil``, ``out.domain``).

        Parameters
        ----------
        backend : {"matplotlib", "plotly"}, optional
            Draw with Matplotlib, or as an interactive Plotly figure (in ``result.fig``; needs
            plotly, see :mod:`plasma_plots.plotly_backend`). Default: the one set with
            :func:`plasma_plots.set_backend`, ``"matplotlib"`` unless changed.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn lines.

        See Also
        --------
        plasma_plots.plotting.plot_equilibrium_profile : The function behind this method.

        Examples
        --------
        >>> out.plot.equilibrium()
        """
        from .plotting import plot_equilibrium_profile

        return plot_equilibrium_profile(self._output.equil, self._output.domain, ax=ax)

    def equilibrium_3d(self, *, scalars: str = "p0", cmap="viridis"):
        """Create a PyVista view of this run's fluid equilibrium; call ``.show()`` on the returned plotter.

        Returns
        -------
        pyvista.Plotter
            The scene, not yet shown.

        See Also
        --------
        plasma_plots.plotting.show_equilibrium : The function behind this method.

        Examples
        --------
        >>> out.plot.equilibrium_3d(scalars="p0").show()
        """
        from .plotting import show_equilibrium

        return show_equilibrium(
            self._output.equil, self._output.domain, scalars=scalars, cmap=cmap
        )

    def domain_3d(
        self, *, n1: int = 8, n2: int = 32, n3: int = 32, surface: bool = True
    ):
        """Draw a PyVista wireframe of this run's mapping (``out.domain``); call ``.show()`` on it.

        Returns
        -------
        pyvista.Plotter
            The scene, not yet shown.

        See Also
        --------
        plasma_plots.pyvista_plots.pyvista_domain : The function behind this method.

        Examples
        --------
        >>> out.plot.domain_3d().show()
        >>> out.plot.domain_3d(n3=1).show()
        """
        from .pyvista_plots import pyvista_domain

        return pyvista_domain(self._output.domain, n1=n1, n2=n2, n3=n3, surface=surface)

    @property
    def profile(self) -> "ProfilePlots":
        """Plots of this run's timing regions, e.g. ``out.plot.profile.gantt()``.

        Needs the optional ``scope-profiler`` extra (``pip install "plasma-plots[profiling]"``)
        and a run recorded with ``sim.run(profiling_activated=True)``.

        Returns
        -------
        ProfilePlots
            The profiling plots of this run.
        """
        return ProfilePlots(self._output)

    def __call__(self, *args, **kwargs):
        """Show the quick default plot: an overview of every scalar time series.

        Parameters
        ----------
        *args
            Passed on to :meth:`scalars`.
        **kwargs
            Passed on to :meth:`scalars`.

        Returns
        -------
        PlotResult
            The figure, the axes and the drawn lines.

        See Also
        --------
        OutputPlots.scalars : The method behind this call.

        Examples
        --------
        >>> out.plot()
        >>> out.plot(logy=True)
        """
        return self.scalars(*args, **kwargs)


def _quadrature_grid(output):
    """Gauss-Legendre points (degree + 1 per element) and weights of each logical direction."""
    elements = tuple(output.grid.num_elements)
    degrees = tuple(output.derham_opts.degree)
    etas, weights = {}, {}
    for dim, n_elements, degree in zip(("eta1", "eta2", "eta3"), elements, degrees):
        x, w = np.polynomial.legendre.leggauss(int(degree) + 1)
        starts = np.arange(n_elements)[:, None] / n_elements
        etas[dim] = (starts + (x[None] + 1) / (2 * n_elements)).ravel()
        weights[dim] = np.tile(w / (2 * n_elements), n_elements)
    return etas, weights


def _matching_quadrature(field, etas, weights):
    """The Gauss weights if ``field`` sits on the Gauss grid, else ``None`` (default rules)."""
    for dim in ("eta1", "eta2", "eta3"):
        if dim not in field.dims or field.sizes[dim] != etas[dim].size:
            return None
        if not np.allclose(np.asarray(field[dim], dtype=float), etas[dim]):
            return None
    return weights


def _linear_mhd_energies(output, velocity, b_field, pressure, gamma):
    import xarray as xr

    from .analysis import field_energy, volume_integral

    etas_q, weights_q = _quadrature_grid(output)

    def array(product, representation):
        # a raw FEEC field in its own space's representation, at the Gauss points of every
        # element (the default products are in "norm" / "0" representations, on one point per
        # cell, which neither the mass matrices nor an exact quadrature use)
        if isinstance(product, str):
            return output.evaluate(
                product,
                eta1=etas_q["eta1"],
                eta2=etas_q["eta2"],
                eta3=etas_q["eta3"],
                representation=representation,
            )
        return product

    def etas(field):
        return [np.asarray(field[d], dtype=float) for d in ("eta1", "eta2", "eta3")]

    domain, equil = output.domain, output.equil
    if not hasattr(equil, "_domain"):
        equil.domain = (
            domain  # the equilibrium profiles are pulled back to this run's mapping
        )
    fields = {
        name: array(value, representation)
        for name, value, representation in (
            ("u", velocity, "2"),
            ("b", b_field, "2"),
            ("p", pressure, "3"),
        )
        if value is not None
    }
    energies = {}
    if "u" in fields:
        n0 = np.asarray(equil.n0(*etas(fields["u"])), dtype=float)
        energies["en_U"] = field_energy(
            fields["u"],
            form=2,
            weight=n0,
            domain=domain,
            quadrature=_matching_quadrature(fields["u"], etas_q, weights_q),
        )
    if "b" in fields:
        energies["en_B"] = field_energy(
            fields["b"],
            form=2,
            domain=domain,
            quadrature=_matching_quadrature(fields["b"], etas_q, weights_q),
        )
    if "p" in fields:
        p0 = np.asarray(equil.p0(*etas(fields["p"])), dtype=float)
        with np.errstate(divide="ignore"):
            inverse = np.where(p0 > 1e-12 * p0.max(), 1.0 / p0, np.nan)
        energies["en_thermal"] = field_energy(
            fields["p"],
            form=3,
            weight=inverse,
            domain=domain,
            normalization=1.0 / gamma,
            quadrature=_matching_quadrature(fields["p"], etas_q, weights_q),
        )
        energies["en_p"] = volume_integral(
            fields["p"],
            form=3,
            domain=domain,
            quadrature=_matching_quadrature(fields["p"], etas_q, weights_q),
        ) / (gamma - 1)
    if not energies:
        raise ValueError("pass at least one of velocity, b_field and pressure")
    quadratic = [energies[k] for k in ("en_U", "en_B", "en_thermal") if k in energies]
    energies["en_tot"] = sum(quadratic[1:], quadratic[0])
    for name, values in energies.items():
        values.attrs = {**values.attrs, "label": name}
    return xr.Dataset(energies, attrs={"label": "LinearMHD energies from fields"})


class OutputAnalysis:
    """Spectral diagnostics of a run's products, as ``out.analysis.<kind>(product, ...)``.

    ``product`` is a product name (``"mhd/velocity"``, evaluated with ``out.evaluate``) or an
    already selected ``xarray.DataArray``. Select a component, slice or time interval before
    transforming when you do not need the whole field: the selected values are loaded into
    memory. The same diagnostics, and more, are available on any array as
    ``array.plasma.analysis.<kind>(...)``; see :mod:`plasma_plots.spectral`.

    Parameters
    ----------
    output : Output
        The struphy run whose products are analyzed. Usually reached as ``out.analysis``.

    Examples
    --------
    >>> spectrum = out.analysis.time_fft(phi.isel(eta2=0, eta3=0))
    >>> modes = out.analysis.mode_spectrum("em_fields/phi_log")
    """

    def __init__(self, output: "Output"):
        self._output = output

    def _array(self, product):
        return self._output.evaluate(product) if isinstance(product, str) else product

    def fft(
        self, product, *, dim: str, detrend: bool = False, window: str | None = None
    ):
        """Compute two-sided Fourier coefficients of a product along ``dim``.

        Parameters
        ----------
        product : str or xarray.DataArray
            A product name (e.g. ``"mhd/velocity"``), evaluated with ``out.evaluate``, or an
            already selected array.

        Returns
        -------
        xarray.DataArray
            Complex coefficients, with ``dim`` replaced by ``omega`` (time) or ``k_<dim>``.

        See Also
        --------
        plasma_plots.spectral.fft : The function behind this method.

        Examples
        --------
        >>> out.analysis.fft(phi.isel(t=-1, eta2=0, eta3=0), dim="eta1")
        """
        from .spectral import fft

        return fft(self._array(product), dim=dim, detrend=detrend, window=window)

    def time_fft(self, product, *, detrend: bool = False, window: str | None = None):
        """Compute one-sided temporal Fourier coefficients and per-bin power of a product.

        Parameters
        ----------
        product : str or xarray.DataArray
            A product name (e.g. ``"mhd/velocity"``), evaluated with ``out.evaluate``, or an
            already selected array.

        Returns
        -------
        xarray.Dataset
            The complex coefficients and the power over ``omega``.

        See Also
        --------
        plasma_plots.spectral.time_fft : The function behind this method.

        Examples
        --------
        >>> out.analysis.time_fft(phi.isel(eta1=0.5, eta2=0, eta3=0), window="hann")
        """
        from .spectral import time_fft

        return time_fft(self._array(product), detrend=detrend, window=window)

    def filter_time(
        self, product, *, dims=None, omega_min: float = 1e-8, pad_bins: int = 0
    ):
        """Reconstruct the dominant temporal frequency band of a product.

        Parameters
        ----------
        product : str or xarray.DataArray
            A product name (e.g. ``"mhd/velocity"``), evaluated with ``out.evaluate``, or an
            already selected array.

        Returns
        -------
        TimeFilterResult
            The filtered field and the reduced spectrum with the selected band.

        See Also
        --------
        plasma_plots.spectral.filter_time : The function behind this method.

        Examples
        --------
        >>> result = out.analysis.filter_time(phi)
        >>> result.filtered.plasma.plot.slice(t=-1, eta3=0)
        """
        from .spectral import filter_time

        return filter_time(
            self._array(product), dims=dims, omega_min=omega_min, pad_bins=pad_bins
        )

    def linear_mhd_energies(
        self,
        *,
        velocity="mhd/velocity",
        b_field="em_fields/b_field",
        pressure="mhd/pressure",
        gamma: float = 5 / 3,
    ):
        """Compute LinearMHD's energy scalars from fields, as a Dataset of time series.

        The scalars are ``en_U``, ``en_B``, ``en_thermal``, ``en_p`` and
        ``en_tot = en_U + en_B + en_thermal``. Same definitions as the scalars saved during the run
        (``en_U = 1/2 u^T M2n u``, ...),
        with the run's ``domain`` and equilibrium (``n0``, ``p0``), but evaluated by quadrature
        on the post-processing grid, so they also work for fields that were never simulated:
        pass a filtered array (e.g. ``filter_time(...).filtered``) to get the energy in one mode.
        Each argument is a raw field name (evaluated at the Gauss points of
        :meth:`quadrature_grid`, in its FEEC space's own representation), an array in that
        representation (on the Gauss grid its weights are exact; elsewhere see
        :func:`~plasma_plots.analysis.quadrature_weights`), or ``None`` to skip it: 2-form
        components for ``velocity`` and ``b_field`` (``out.evaluate("mhd/velocity",
        representation="2")``), a 3-form for ``pressure`` (``representation="3"``). The default
        post-processing products are in other representations (``"norm"``, ``"0"``) and would
        give wrong energies. Points where ``p0`` vanishes are left out of ``en_thermal``.

        Parameters
        ----------
        velocity : str, xarray.DataArray or None, optional
            The velocity as 2-form components (``en_U``, weighted by ``n0``).
            Default: ``"mhd/velocity"``.
        b_field : str, xarray.DataArray or None, optional
            The magnetic field as 2-form components (``en_B``). Default: ``"em_fields/b_field"``.
        pressure : str, xarray.DataArray or None, optional
            The pressure as a 3-form (``en_thermal``, weighted by ``1/p0``, and ``en_p``).
            Default: ``"mhd/pressure"``.
        gamma : float, optional
            The adiabatic index: ``en_thermal`` is normalized by ``1/gamma`` and
            ``en_p = ∫ p / (gamma - 1)``. Default: ``5/3``.

        Returns
        -------
        xarray.Dataset
            One time series per computed scalar (only those whose fields were given), plus
            ``en_tot``.

        Raises
        ------
        ValueError
            If ``velocity``, ``b_field`` and ``pressure`` are all ``None``.

        See Also
        --------
        quadrature_grid : The Gauss points the raw fields are evaluated at.
        plasma_plots.analysis.field_energy : The energy integral of one field.

        Examples
        --------
        >>> energies = out.analysis.linear_mhd_energies()
        >>> filtered = out.analysis.filter_time(
        ...     out.evaluate("mhd/velocity", representation="2")
        ... ).filtered
        >>> out.analysis.linear_mhd_energies(
        ...     velocity=filtered, b_field=None, pressure=None
        ... )
        """
        return _linear_mhd_energies(self._output, velocity, b_field, pressure, gamma)

    def quadrature_grid(self):
        """Return the run's Gauss-Legendre quadrature points and weights in each logical direction.

        The points (spline degree + 1 per element) and their weights for each of ``eta1``,
        ``eta2``, ``eta3`` form the grid on which spline fields squared integrate exactly.
        Evaluate a field there (``out.evaluate(name, eta1=etas["eta1"], ...,
        representation="2")``), filter it, and ``linear_mhd_energies`` or ``field_energy(...,
        quadrature=weights)`` give its energy as the run's own scalars would.

        Returns
        -------
        etas : dict of str to numpy.ndarray
            The points in ``[0, 1]`` for ``"eta1"``, ``"eta2"``, ``"eta3"``.
        weights : dict of str to numpy.ndarray
            The matching quadrature weights, summing to 1 in each direction.

        Examples
        --------
        >>> etas, weights = out.analysis.quadrature_grid()
        >>> b = out.evaluate(
        ...     "em_fields/b_field",
        ...     eta1=etas["eta1"],
        ...     eta2=etas["eta2"],
        ...     eta3=etas["eta3"],
        ...     representation="2",
        ... )
        """
        return _quadrature_grid(self._output)

    def mode_spectrum(
        self, product, *, dims=("eta2", "eta3"), names=("m", "n"), periods=1.0
    ):
        """Compute complex amplitudes of a product over poloidal/toroidal mode numbers.

        Parameters
        ----------
        product : str or xarray.DataArray
            A product name (e.g. ``"mhd/velocity"``), evaluated with ``out.evaluate``, or an
            already selected array.

        Returns
        -------
        xarray.DataArray
            Complex amplitudes over the mode numbers ``names`` and every remaining dimension.

        See Also
        --------
        plasma_plots.spectral.mode_spectrum : The function behind this method.

        Examples
        --------
        >>> out.analysis.mode_spectrum("em_fields/phi_log")
        >>> out.analysis.mode_spectrum(phi.isel(t=-1), dims="eta3", names="n")
        """
        from .spectral import mode_spectrum

        return mode_spectrum(
            self._array(product), dims=dims, names=names, periods=periods
        )


_OUTPUT_MODULE = "struphy.post_processing.output"


def _register_output_plot_property(module=None):
    """Wire ``out.plot`` and ``out.analysis`` to :class:`OutputPlots` and :class:`OutputAnalysis`.

    ``module`` is struphy's ``struphy.post_processing.output``, already imported: plasma-plots
    never imports struphy itself, which takes seconds (see :class:`_RegisterOnImport`).
    """
    module = module if module is not None else sys.modules.get(_OUTPUT_MODULE)
    Output = getattr(module, "Output", None)
    if Output is None:
        return
    # never replace an attribute struphy defines itself (e.g. a future Output.analysis method)
    for name, accessor in (("plot", OutputPlots), ("analysis", OutputAnalysis)):
        if name not in Output.__dict__:
            setattr(Output, name, property(accessor))


class _RegisterOnImport(importlib.abc.MetaPathFinder):
    """Register ``out.plot`` as soon as struphy's output module is imported, in whichever order
    struphy and plasma-plots are imported, without importing struphy in a process that never
    uses it (e.g. plotting GENE output). struphy itself imports plasma-plots from
    ``Output.__init__``, which is too late when ``plasma_plots`` was imported first."""

    def find_spec(self, name, path=None, target=None):
        if name != _OUTPUT_MODULE:
            return None
        for finder in sys.meta_path:  # the spec the other finders would give
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if loader is None or not hasattr(loader, "exec_module"):
            return spec
        execute = loader.exec_module

        def exec_module(module):
            execute(module)
            _register_output_plot_property(module)

        loader.exec_module = exec_module
        sys.meta_path.remove(self)  # needed once
        return spec


def _register_output_plot_property_when_available():
    if _OUTPUT_MODULE in sys.modules:
        _register_output_plot_property()
    elif not any(isinstance(finder, _RegisterOnImport) for finder in sys.meta_path):
        sys.meta_path.insert(0, _RegisterOnImport())


def _complete_docstrings():
    """Give every method the parameter docs it inherits, so ``help()`` shows them all."""
    from ._docs import add_menu, complete_class

    for cls in (OutputPlots, OutputAnalysis, ProfilePlots):
        complete_class(cls)
        add_menu(cls)


_complete_docstrings()
_register_output_plot_property_when_available()
