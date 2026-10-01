"""Plots and diagnostics of labeled xarray data from plasma simulations, such as Struphy's output.

plasma-plots adds accessors to the objects Struphy returns; you rarely call a function directly:

* ``out.plot`` and ``out.analysis`` on a Struphy ``Output``: whole-run plots and diagnostics.
* ``array.plasma.plot``, ``.analysis`` and ``.data`` on every ``xarray.DataArray``, e.g. a product
  from ``out.evaluate("em_fields/phi")``.
* ``dataset.plasma.plot``, ``.analysis`` and ``.data`` on marker Datasets such as orbits.

Creating a Struphy ``Output`` loads plasma-plots, so its output needs no import. For xarray data
from elsewhere, ``import plasma_plots`` first; without it, ``.plasma`` raises
``AttributeError: 'DataArray' object has no attribute 'plasma'``.

Printing an accessor lists its methods (``print(phi.plasma.plot)``), and ``help()`` on a method
shows every parameter (``help(phi.plasma.plot.slice)``). ``plasma-plots guide`` (or
``python -m plasma_plots guide``) prints this guide, ``plasma-plots api`` an index of every method and
function with its signature. The ``plasma-plots`` command also saves figures from the shell, e.g.
``plasma-plots plot sim_1 em_fields/phi slice t=-1 eta3=0 -o phi.png`` or
``plasma-plots quicklook sim_1 -o figures/`` (``plasma-plots --help``).

A post-processing pipeline
--------------------------
>>> from struphy import Output  # doctest: +SKIP
>>> out = Output("sim_1")  # the run's output folder
>>> out.plot.energies().save("energies.png")  # the energy budget and its drift
>>> # dims (t, eta1, eta2, eta3), coordinates X, Y, Z
>>> phi = out.evaluate("em_fields/phi")
>>> phi.plasma.plot.slice(coords="physical", plane="XY", t=-1, eta3=0).save(
...     "phi.png"
... )
>>> phi.plasma.plot.animation(x="eta1", y="eta2", eta3=0).save(
...     "phi.gif", writer="pillow"
... )
>>> # complex amplitudes over mode numbers m, n
>>> modes = phi.plasma.analysis.mode_spectrum()
>>> # the strongest (m, n) over time
>>> phi.plasma.plot.mode_amplitudes(top=4, fit=True)
>>> # guiding-center orbits, by orbit class
>>> out.kinetic_ions.orbits.plasma.plot.poloidal()

Selecting what to show
----------------------
Name every dimension a plot doesn't draw: an integer is a position (``t=-1`` the last, ``t=0`` the
first) and a float the nearest coordinate value (``t=0.35``). Slices draw two dimensions: logical
ones (``x="eta1", y="eta2"``), or physical ones with ``coords="physical", plane="XY"`` (or
``"XZ"``, ``"YZ"``, ``"RZ"``). ``array.plasma.data.<plot>(...)`` returns the selected data a plot
would draw, instead of the figure.

What is where
-------------
* Time series and rates: ``plot.timeseries(fit=(t0, t1), reference=...)``,
  ``analysis.growth_rate``, ``analysis.damping_rate``, ``analysis.envelope``,
  ``analysis.oscillation_frequency`` (from zero crossings).
* Profiles: ``plot.lineout``, ``plot.profiles``, ``plot.line_animation`` (``alongside=`` for
  panels in sync), each with ``reference=`` for exact solutions; ``analysis.map_coordinate`` for
  physical coordinates (``eta1`` → ``r`` in m).
* Several plots in one figure: ``with plasma_plots.figure(2, 1) as fig:`` and ``ax=fig[0]``,
  ``ax=fig[1]``.
* 2-D fields: ``plot.slice``, ``plot.panels``, ``plot.animation``, ``plot.viewer``,
  ``plot.frames``, with ``levels=`` (contour lines), ``overlays=`` (a second field's contours,
  boundary, lines, points), ``symmetric=``, ``robust=``; ``plot.vector`` for vector fields.
* 3-D views (``pip install "plasma-plots[pyvista]"``): ``plot.isosurface``, ``plot.slices_3d``,
  ``plot.glyphs``, ``plot.streamlines``, ``plot.movie``; ``data.to_vtk`` for ParaView.
* Spectra: ``analysis.time_fft``, ``analysis.spectral_peaks``, ``analysis.filter_time``,
  ``analysis.mode_spectrum``, ``analysis.matrix_pencil``, ``analysis.cross_spectrum``;
  ``plot.power_spectrum``, ``plot.spectrogram``, ``plot.mode_amplitudes``, ``plot.mode_profiles``.
* Dispersion relations: ``plot.dispersion(branches=...)``, ``analysis.dispersion`` and
  ``.plasma.analysis.trace_branch(theory)`` on the spectrum.
* Comparing with theory: ``analysis.error(exact)``, ``plot.against_theory(theory)``,
  ``analysis.project_mode``; convergence studies with ``plot.convergence``.
* Vector calculus on mapped domains: ``analysis.gradient``, ``analysis.divergence``,
  ``analysis.curl``, ``analysis.flux_function``, ``analysis.toroidal_components``.
* Particles: ``dataset.plasma.plot.scatter``, ``.animation`` (``trail=``, ``paths=``,
  ``color="classification"``), ``.paths``, ``.poloidal``,
  ``.orbit_grid``, ``.orbit_classification``, ``.orbits_3d``; ``dataset.plasma.analysis.
  classify_orbits``, ``.orbit_invariants``, ``.bounce_period``; binned distributions with
  ``plot.slice(x="eta1", y="v1")`` and ``analysis.velocity_moments``.
* Whole runs: ``out.plot.energies``, ``out.plot.scalars``, ``out.plot.equilibrium``,
  ``out.plot.profile``; ``out.analysis.linear_mhd_energies``, ``out.analysis.time_fft``,
  ``out.analysis.mode_spectrum``.
* Integrals: ``plasma_plots.analysis.volume_integral`` and ``field_energy``;
  ``analysis.surface_average`` for flux-surface averages.
* GVEC equilibria: ``.plasma`` reads ``state.evaluate(...)`` itself, ``plasma_plots.from_gvec(ds)``
  attaches the geometry to every variable; poloidal planes with
  ``overlays={"coordinate_lines": {"rho": 4, "theta_P": 8}}`` (and ``plane="X1X2"``), ι with
  ``plot.lineout(rationals=4)`` and ``analysis.rational_surfaces``.
* DESC equilibria: ``plasma_plots.from_desc(eq, ["|B|", "iota", "sqrt(g)"], rho=11, theta=64,
  zeta=40)`` evaluates them into the same flux-coordinate Datasets (``sfl="pest"`` for the PEST
  angle ``theta_P``); DESC's names stay, ``ev["|B|"].plasma.plot...``.
* Analytic theory to compare with (plain functions, not accessors): ``plasma_plots.theory.kinetic``
  (Landau damping, beam instabilities, Weibel), ``.waves`` (MHD, Hall-MHD and cold-plasma waves,
  drift waves, continua), ``.parameters`` (plasma parameters, Struphy's units), ``.orbits``,
  ``.exact`` (Riemann problem, dam break, diffusion, ...) and ``.numerics`` (time-integrator and
  discretization errors). Their functions go straight into ``branches=``, ``reference=`` and
  ``theory=``, e.g. ``phi.plasma.plot.dispersion(branches={"kinetic": kinetic.langmuir})``.

Plots return a ``PlotResult`` (``.fig``, ``.ax``, ``.save(path)``, ``.show()``); animations a
``matplotlib.animation.FuncAnimation`` (keep a reference; ``.save("a.gif", writer="pillow")``); 3-D
views a ``pyvista.Plotter`` (``.show()``, ``.screenshot(path)``; ``pyvista.OFF_SCREEN = True`` in
scripts). Analysis methods return labeled xarray objects, which have ``.plasma`` in turn.

Every Matplotlib plot also draws as an interactive Plotly figure (``pip install
"plasma-plots[plotly]"``): ``phi.plasma.plot.slice(t=-1, eta3=0, backend="plotly")``, or
``plasma_plots.set_backend("plotly")`` for all of them. The result is a ``PlotResult`` too
(``.save("phi.html")``); animations and viewers get a slider. See
:mod:`plasma_plots.plotly_backend`.

Under MPI (``mpirun -n 4 python script.py``), plots are drawn and saved on rank 0 only; the other
ranks get a ``SkippedPlot`` whose methods do nothing, so one script runs unchanged in serial and
in parallel. Analysis runs on every rank. See :mod:`plasma_plots.mpi`.

The functions behind the accessors, for plain ``xarray.DataArray`` input, are in
``plasma_plots.plotting``, ``.analysis``, ``.spectral``, ``.spectral_plots``, ``.pyvista_plots``
and ``.arrays``.

Guides and the full reference: https://max-models.github.io/plasma-plots (for language models:
https://max-models.github.io/plasma-plots/llms.txt).

Example
-------
Runs as is, on synthetic data:

>>> import matplotlib
>>> matplotlib.use("Agg")
>>> import numpy as np
>>> import xarray as xr
>>> import plasma_plots
>>> t = np.linspace(0.0, 20.0, 201)
>>> x = np.linspace(0.0, 1.0, 64, endpoint=False)
>>> phi = xr.DataArray(
...     0.01 * np.exp(0.1 * t)[:, None] * np.sin(2 * np.pi * x)[None],
...     dims=("t", "eta1"),
...     coords={"t": t, "eta1": x},
...     name="phi",
... )
>>> # the k = 1 amplitude over t
>>> amplitude = phi.plasma.analysis.project_mode(dim="eta1", number=1)
>>> round(
...     float(amplitude.plasma.analysis.growth_rate(window=(5.0, 20.0)).rate),
...     3,
... )
0.1
>>> result = phi.plasma.plot.slice(x="eta1", y="t")  # a space-time map
>>> type(result).__name__
'PlotResult'
"""

from . import \
    output_accessors  # noqa: F401  (registers Output.plot, if struphy is installed)
from .accessors import PlasmaAccessor
from .desc import from_desc
from .figures import figure
from .gvec import from_gvec
from .mpi import SkippedPlot, is_plotting_rank, mpi_rank
from .plotly_backend import get_backend, set_backend
from .plotting import save_figure

__all__ = [
    "SkippedPlot",
    "PlasmaAccessor",
    "figure",
    "from_desc",
    "from_gvec",
    "get_backend",
    "is_plotting_rank",
    "mpi_rank",
    "save_figure",
    "set_backend",
]
