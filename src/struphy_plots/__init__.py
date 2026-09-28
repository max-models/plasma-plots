"""Plots and diagnostics for Struphy output, on labeled xarray data.

struphy-plots adds accessors to the objects Struphy returns; you rarely call a function directly:

* ``out.plot`` and ``out.analysis`` on a Struphy ``Output``: whole-run plots and diagnostics.
* ``array.struphy.plot``, ``.analysis`` and ``.data`` on every ``xarray.DataArray``, e.g. a product
  from ``out.evaluate("em_fields/phi")``.
* ``dataset.struphy.plot``, ``.analysis`` and ``.data`` on marker Datasets such as orbits.

Creating a Struphy ``Output`` loads struphy-plots, so its output needs no import. For xarray data
from elsewhere, ``import struphy_plots`` first; without it, ``.struphy`` raises
``AttributeError: 'DataArray' object has no attribute 'struphy'``.

Printing an accessor lists its methods (``print(phi.struphy.plot)``), and ``help()`` on a method
shows every parameter (``help(phi.struphy.plot.slice)``). ``python -m struphy_plots`` prints this
guide.

A post-processing pipeline
--------------------------
>>> from struphy import Output                     # doctest: +SKIP
>>> out = Output("sim_1")                          # the run's output folder
>>> out.plot.energies().save("energies.png")       # the energy budget and its drift
>>> phi = out.evaluate("em_fields/phi")            # dims (t, eta1, eta2, eta3), coordinates X, Y, Z
>>> phi.struphy.plot.slice(coords="physical", plane="XY", t=-1, eta3=0).save("phi.png")
>>> phi.struphy.plot.animation(x="eta1", y="eta2", eta3=0).save("phi.gif", writer="pillow")
>>> modes = phi.struphy.analysis.mode_spectrum()   # complex amplitudes over mode numbers m, n
>>> phi.struphy.plot.mode_amplitudes(top=4, fit=True)  # the strongest (m, n) over time
>>> out.kinetic_ions.orbits.struphy.plot.poloidal()  # guiding-center orbits, by orbit class

Selecting what to show
----------------------
Name every dimension a plot doesn't draw: an integer is a position (``t=-1`` the last, ``t=0`` the
first) and a float the nearest coordinate value (``t=0.35``). Slices draw two dimensions: logical
ones (``x="eta1", y="eta2"``), or physical ones with ``coords="physical", plane="XY"`` (or
``"XZ"``, ``"YZ"``, ``"RZ"``). ``array.struphy.data.<plot>(...)`` returns the selected data a plot
would draw, instead of the figure.

What is where
-------------
* Time series and rates: ``plot.timeseries(fit=(t0, t1), reference=...)``,
  ``analysis.growth_rate``, ``analysis.damping_rate``, ``analysis.envelope``,
  ``analysis.oscillation_frequency`` (from zero crossings).
* Profiles: ``plot.lineout``, ``plot.profiles``, ``plot.line_animation`` (``alongside=`` for
  panels in sync), each with ``reference=`` for exact solutions; ``analysis.map_coordinate`` for
  physical coordinates (``eta1`` → ``r`` in m).
* Several plots in one figure: ``with struphy_plots.figure(2, 1) as fig:`` and ``ax=fig[0]``,
  ``ax=fig[1]``.
* 2-D fields: ``plot.slice``, ``plot.panels``, ``plot.animation``, ``plot.viewer``,
  ``plot.frames``, with ``levels=`` (contour lines), ``overlays=`` (a second field's contours,
  boundary, lines, points), ``symmetric=``, ``robust=``; ``plot.vector`` for vector fields.
* 3-D views (``pip install "struphy-plots[pyvista]"``): ``plot.isosurface``, ``plot.slices_3d``,
  ``plot.glyphs``, ``plot.streamlines``, ``plot.movie``; ``data.to_vtk`` for ParaView.
* Spectra: ``analysis.time_fft``, ``analysis.spectral_peaks``, ``analysis.filter_time``,
  ``analysis.mode_spectrum``, ``analysis.matrix_pencil``, ``analysis.cross_spectrum``;
  ``plot.power_spectrum``, ``plot.spectrogram``, ``plot.mode_amplitudes``, ``plot.mode_profiles``.
* Dispersion relations: ``plot.dispersion(branches=...)``, ``analysis.dispersion`` and
  ``.struphy.analysis.trace_branch(theory)`` on the spectrum.
* Comparing with theory: ``analysis.error(exact)``, ``plot.against_theory(theory)``,
  ``analysis.project_mode``; convergence studies with ``plot.convergence``.
* Vector calculus on mapped domains: ``analysis.gradient``, ``analysis.divergence``,
  ``analysis.curl``, ``analysis.flux_function``, ``analysis.toroidal_components``.
* Particles: ``dataset.struphy.plot.scatter``, ``.animation`` (``trail=``, ``paths=``,
  ``color="classification"``), ``.paths``, ``.poloidal``,
  ``.orbit_grid``, ``.orbit_classification``, ``.orbits_3d``; ``dataset.struphy.analysis.
  classify_orbits``, ``.orbit_invariants``, ``.bounce_period``; binned distributions with
  ``plot.slice(x="eta1", y="v1")`` and ``analysis.velocity_moments``.
* Whole runs: ``out.plot.energies``, ``out.plot.scalars``, ``out.plot.equilibrium``,
  ``out.plot.profile``; ``out.analysis.linear_mhd_energies``, ``out.analysis.time_fft``,
  ``out.analysis.mode_spectrum``.
* Integrals: ``struphy_plots.analysis.volume_integral`` and ``field_energy``.
* Analytic theory to compare with (plain functions, not accessors): ``struphy_plots.theory.kinetic``
  (Landau damping, beam instabilities, Weibel), ``.waves`` (MHD, Hall-MHD and cold-plasma waves,
  drift waves, continua), ``.parameters`` (plasma parameters, Struphy's units), ``.orbits``,
  ``.exact`` (Riemann problem, dam break, diffusion, ...) and ``.numerics`` (time-integrator and
  discretization errors). Their functions go straight into ``branches=``, ``reference=`` and
  ``theory=``, e.g. ``phi.struphy.plot.dispersion(branches={"kinetic": kinetic.langmuir})``.

Plots return a ``PlotResult`` (``.fig``, ``.ax``, ``.save(path)``, ``.show()``); animations a
``matplotlib.animation.FuncAnimation`` (keep a reference; ``.save("a.gif", writer="pillow")``); 3-D
views a ``pyvista.Plotter`` (``.show()``, ``.screenshot(path)``; ``pyvista.OFF_SCREEN = True`` in
scripts). Analysis methods return labeled xarray objects, which have ``.struphy`` in turn.

Every Matplotlib plot also draws as an interactive Plotly figure (``pip install
"struphy-plots[plotly]"``): ``phi.struphy.plot.slice(t=-1, eta3=0, backend="plotly")``, or
``struphy_plots.set_backend("plotly")`` for all of them. The result is a ``PlotResult`` too
(``.save("phi.html")``); animations and viewers get a slider. See
:mod:`struphy_plots.plotly_backend`.

Under MPI (``mpirun -n 4 python script.py``), plots are drawn and saved on rank 0 only; the other
ranks get a ``SkippedPlot`` whose methods do nothing, so one script runs unchanged in serial and
in parallel. Analysis runs on every rank. See :mod:`struphy_plots.mpi`.

The functions behind the accessors, for plain ``xarray.DataArray`` input, are in
``struphy_plots.plotting``, ``.analysis``, ``.spectral``, ``.spectral_plots``, ``.pyvista_plots``
and ``.arrays``.

Guides and the full reference: https://struphy-hub.github.io/struphy-plots (for language models:
https://struphy-hub.github.io/struphy-plots/llms.txt).

Example
-------
Runs as is, on synthetic data:

>>> import matplotlib
>>> matplotlib.use("Agg")
>>> import numpy as np
>>> import xarray as xr
>>> import struphy_plots
>>> t = np.linspace(0.0, 20.0, 201)
>>> x = np.linspace(0.0, 1.0, 64, endpoint=False)
>>> phi = xr.DataArray(
...     0.01 * np.exp(0.1 * t)[:, None] * np.sin(2 * np.pi * x)[None],
...     dims=("t", "eta1"), coords={"t": t, "eta1": x}, name="phi",
... )
>>> amplitude = phi.struphy.analysis.project_mode(dim="eta1", number=1)   # the k = 1 amplitude over t
>>> round(float(amplitude.struphy.analysis.growth_rate(window=(5.0, 20.0)).rate), 3)
0.1
>>> result = phi.struphy.plot.slice(x="eta1", y="t")                       # a space-time map
>>> type(result).__name__
'PlotResult'
"""

from . import output_accessors  # noqa: F401  (registers Output.plot, if struphy is installed)
from .accessors import StruphyAccessor
from .mpi import SkippedPlot, is_plotting_rank, mpi_rank
from .figures import figure
from .plotly_backend import get_backend, set_backend
from .plotting import save_figure

__all__ = [
    "SkippedPlot",
    "StruphyAccessor",
    "figure",
    "get_backend",
    "is_plotting_rank",
    "mpi_rank",
    "save_figure",
    "set_backend",
]
