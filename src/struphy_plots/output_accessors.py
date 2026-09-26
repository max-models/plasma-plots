"""Optional plots that need a whole run.

Plots and diagnostics of a single array live on the array, see
:class:`~struphy.post_processing.xarray_accessors.StruphyAccessor`:
``out.em_fields.phi_log.struphy.plot.slice(...)``, or from a value returned by
``out.evaluate("em_fields/phi_log")``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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

    def scalars(self, names=None, *, relative_to: str | None = None, logy: bool = False):
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

    def energies(
        self,
        *,
        parts=None,
        total: str | None = "en_tot",
        groups: dict | None = None,
        logy: bool = False,
    ):
        """The run's energy budget: its ``en_*`` scalars, the relative drift of ``total``, and,
        with ``groups`` (e.g. ``{"wave": ["en_U", "en_B", "en_p"], "energetic ions": ["en_fv",
        "en_fB"]}``), the energy exchanged between them. See
        :func:`struphy_plots.plotting.plot_energy_budget`."""
        from .plotting import plot_energy_budget

        return plot_energy_budget(
            self._output.scalars,
            parts=parts,
            total=total,
            groups=groups,
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

        return show_equilibrium(self._output.equil, self._output.domain, scalars=scalars, cmap=cmap)

    def domain_3d(self, *, n1: int = 8, n2: int = 32, n3: int = 32, surface: bool = True):
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

    def fft(self, product, *, dim: str, detrend: bool = False, window: str | None = None):
        """Two-sided Fourier coefficients along ``dim``; see :func:`struphy_plots.spectral.fft`."""
        from .spectral import fft

        return fft(self._array(product), dim=dim, detrend=detrend, window=window)

    def time_fft(self, product, *, detrend: bool = False, window: str | None = None):
        """One-sided temporal coefficients and per-bin power; see
        :func:`struphy_plots.spectral.time_fft`."""
        from .spectral import time_fft

        return time_fft(self._array(product), detrend=detrend, window=window)

    def filter_time(self, product, *, dims=None, omega_min: float = 1e-8, pad_bins: int = 0):
        """Reconstruct the dominant temporal band; see :func:`struphy_plots.spectral.filter_time`."""
        from .spectral import filter_time

        return filter_time(self._array(product), dims=dims, omega_min=omega_min, pad_bins=pad_bins)

    def linear_mhd_energies(
        self,
        *,
        velocity="mhd/velocity",
        b_field="em_fields/b_field",
        pressure="mhd/pressure",
        gamma: float = 5 / 3,
    ):
        """LinearMHD's energy scalars computed from fields: ``en_U``, ``en_B``, ``en_thermal``,
        ``en_p`` and ``en_tot = en_U + en_B + en_thermal``, as a Dataset of time series.

        Same definitions as the scalars saved during the run (``en_U = 1/2 u^T M2n u``, ...),
        with the run's ``domain`` and equilibrium (``n0``, ``p0``), but evaluated by quadrature
        on the post-processing grid, so they also work for fields that were never simulated:
        pass a filtered array (e.g. ``filter_time(...).filtered``) to get the energy in one mode.
        Each argument is a raw field name (evaluated at the Gauss points of
        :meth:`quadrature_grid`, in its FEEC space's own representation), an array in that
        representation (on the Gauss grid its weights are exact; elsewhere see
        :func:`~struphy_plots.analysis.quadrature_weights`), or ``None`` to skip it: 2-form
        components for ``velocity`` and ``b_field`` (``out.evaluate("mhd/velocity",
        representation="2")``), a 3-form for ``pressure`` (``representation="3"``). The default
        post-processing products are in other representations (``"norm"``, ``"0"``) and would
        give wrong energies. Points where ``p0`` vanishes are left out of ``en_thermal``.
        """
        return _linear_mhd_energies(self._output, velocity, b_field, pressure, gamma)

    def quadrature_grid(self):
        """``(etas, weights)``: Gauss-Legendre points (spline degree + 1 per element) and their
        weights for each of ``e1``, ``e2``, ``e3`` -- the grid on which spline fields squared
        integrate exactly. Evaluate a field there (``out.evaluate(name, eta1=etas["e1"], ...,
        representation="2")``), filter it, and ``linear_mhd_energies`` or ``field_energy(...,
        quadrature=weights)`` give its energy as the run's own scalars would."""
        return _quadrature_grid(self._output)

    def mode_spectrum(self, product, *, dims=("e2", "e3"), names=("m", "n"), periods=1.0):
        """Complex amplitudes over poloidal/toroidal mode numbers; see
        :func:`struphy_plots.spectral.mode_spectrum`."""
        from .spectral import mode_spectrum

        return mode_spectrum(self._array(product), dims=dims, names=names, periods=periods)


def _quadrature_grid(output):
    """Gauss-Legendre points (degree + 1 per element) and weights of each logical direction."""
    elements = tuple(output.grid.num_elements)
    degrees = tuple(output.derham_opts.degree)
    etas, weights = {}, {}
    for dim, n_elements, degree in zip(("e1", "e2", "e3"), elements, degrees):
        x, w = np.polynomial.legendre.leggauss(int(degree) + 1)
        starts = np.arange(n_elements)[:, None] / n_elements
        etas[dim] = (starts + (x[None] + 1) / (2 * n_elements)).ravel()
        weights[dim] = np.tile(w / (2 * n_elements), n_elements)
    return etas, weights


def _matching_quadrature(field, etas, weights):
    """The Gauss weights if ``field`` sits on the Gauss grid, else ``None`` (default rules)."""
    for dim in ("e1", "e2", "e3"):
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
                eta1=etas_q["e1"],
                eta2=etas_q["e2"],
                eta3=etas_q["e3"],
                representation=representation,
            )
        return product

    def etas(field):
        return [np.asarray(field[d], dtype=float) for d in ("e1", "e2", "e3")]

    domain, equil = output.domain, output.equil
    if not hasattr(equil, "_domain"):
        equil.domain = domain  # the equilibrium profiles are pulled back to this run's mapping
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


def _register_output_plot_property():
    """Wire ``out.plot`` to :class:`OutputPlots`, if struphy is installed.

    Guarded so importing ``struphy_plots`` stays optional: the array-level accessor works on any
    labeled ``xarray`` object without struphy installed at all.
    """
    try:
        from struphy.post_processing.output import Output
    except ImportError:
        return
    # never replace an attribute struphy defines itself (e.g. a future Output.analysis method)
    for name, accessor in (("plot", OutputPlots), ("analysis", OutputAnalysis)):
        if name not in Output.__dict__:
            setattr(Output, name, property(accessor))


_register_output_plot_property()
