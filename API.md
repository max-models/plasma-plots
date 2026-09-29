# plasma-plots API index

Plots and diagnostics of labeled xarray data from plasma simulations (Struphy, and any code
whose output follows the same conventions, e.g. GENE). Generated from the code by
`python -m plasma_plots api`: every accessor method and public function, with its signature
and one-line summary. `help()` on any of them shows every parameter; the guides with figures are
at https://struphy-hub.github.io/plasma-plots.

## Conventions every method shares

- **Accessors, not functions.** `import plasma_plots` adds `.plasma` to every
  `xarray.DataArray` and `xarray.Dataset`: `array.plasma.plot.*`, `.analysis.*`, `.data.*`. A
  Struphy `Output` also gets `out.plot.*` and `out.analysis.*` (and loads plasma-plots itself).
- **Selection by keyword.** Name every dimension a plot does not draw: an integer is a position
  (`t=-1` the last, `t=0` the first), a float the nearest coordinate value (`t=0.35`).
  Unknown names, strings and bools raise `TypeError`.
- **Dimensions and attributes.** Time is `t`; logical space `eta1`, `eta2`, `eta3` (mapped
  coordinates `X`, `Y`, `Z` as 2-D/3-D coordinates); vector components `component`; markers
  `marker`. Labels come from `attrs["label"]` (mathtext), units from `attrs["units"]`,
  coordinate labels from their `label`/`long_name`/`units`; `attrs["run"]` titles a figure,
  `attrs["run_name"]` names a run in comparisons.
- **GVEC.** `.plasma` reads GVEC's evaluations (`state.evaluate(...)`) itself: the dimensions
  become `rho`, `theta`/`theta_B`/`theta_P`, `zeta`/`zeta_B` (angles in radians, with a
  `period`), `pos` the `X`, `Y`, `Z` coordinates, `xyz` the `component`; call
  `plasma_plots.from_gvec(ds)` once on the Dataset to give every variable its geometry.
- **DESC.** `plasma_plots.from_desc(eq, names, rho=, theta=, zeta=, sfl=None|"pest")` evaluates a
  DESC equilibrium into a Dataset over `rho`, `theta`/`theta_P`, `zeta` with `X`, `Y`, `Z`
  coordinates and Cartesian vectors; variables keep DESC's names (`ev["|B|"]`), `"sqrt(g)"` is
  the Jacobian.
- **Return values.** Plots return a `PlotResult` (`.fig`, `.ax`, `.artists`, `.fit_results`,
  `.data`, `.save(path)`, `.show()`, `.to_plotly()`); Matplotlib animations a `FuncAnimation`;
  PyVista views a `pyvista.Plotter`; analysis methods labeled xarray objects (which have
  `.plasma` again) or small result dataclasses (`FitResult`, `OscillationFit`, ...).
  `array.plasma.data.<plot>(...)` returns the data a plot would draw, without drawing it.
- **Backends.** Every Matplotlib plot takes `backend="plotly"` for an interactive Plotly figure
  (in a `PlotResult`; animations get a slider); `plasma_plots.set_backend("plotly")` sets the
  default. `ax=` draws into your own axes (Matplotlib only); `plasma_plots.figure(rows, cols)`
  composes several plots into one figure, with either backend.
- **MPI.** Under `mpirun`, plots draw and save on rank 0 only; other ranks get a `SkippedPlot`
  whose methods do nothing. Analysis runs on every rank.
- **Theory.** `plasma_plots.theory.*` are plain functions (complex ω for dispersion relations)
  that go straight into `branches=`, `reference=` and `theory=` of the plots.

## Parameters that mean the same everywhere

- **2-D slices** (`slice`, `panels`, `animation`, `viewer`, `frames`, `view`) draw the two
  dimensions `x=` and `y=` (logical, e.g. `x="eta1", y="eta2"`, or `x="eta1", y="t"` for a
  space-time map), or with `coords="physical", plane="XY"` the mapped coordinates; `plane` is
  one of `"XY"`, `"XZ"`, `"YZ"`, `"RZ"` (`"RZ"`: R = √(X² + Y²), the poloidal plane of a torus).
  `symmetric=True` centres the colours on zero, `robust=True` clips outliers, `levels=` adds
  contour lines, `overlays={...}` a second field's contours, the boundary, lines and points;
  `xlabel=`, `ylabel=`, `colorbar_label=` replace the labels.
- **Fits on plots** take a time window, not a fit: `plot.timeseries(fit=(t0, t1))` (or `fit=True`
  for all), `fit_amplitude=True` for a squared quantity such as an energy; the fit is then in
  `result.fit_results[0]`. **Fits in analysis** take `window=(t0, t1)`:
  `analysis.growth_rate(window=(t0, t1)).rate`.
- **References and theory:** `reference=` (a function of `t` or `x`, or of `x` and `t`, an array,
  an `(x, y)` pair, or a dict of labels to these), `branches={"label": omega_of_k}`,
  `theory=` of `against_theory`.
- **Animations:** `step=n` keeps every n-th frame, `max_frames=n` at most n evenly spaced ones;
  `interval=` is the delay in ms. With `backend="plotly"` they return a `PlotResult` with a slider;
  `result.save("movie.png", frame=k)` saves frame k as the image.
- **Markers** (`dataset.plasma.plot.*` of orbits or particles): `x=`/`y=` name variables of the
  Dataset (`x="R"` is √(x² + y²) if not a variable), `color=` a variable name or
  `"classification"` (passing/trapped/lost, needs `v_par`); animations take `trail=n`,
  `paths=True`, `background=` a field.
- **Composing:** `with plasma_plots.figure(2, 1, sharex=True) as fig:` then `ax=fig[0]`,
  `ax=fig[1]` (flat index, or `fig[row, col]`); `fig.save(...)` after the block (inside it,
  it saves the panels drawn so far).

## Accessor methods

### array.plasma.plot

Plots of one array, as array.plasma.plot.<kind>(...).

- `array.plasma.plot.timeseries(*others, logy=True, fit=None, fit_amplitude=False, title=None, ax=None, reference=None, backend=None)`: Plot this time series, and any others given, in one axes.
  - e.g. `energy.plasma.plot.timeseries(logy=True, fit=(0.0, 2.0), fit_amplitude=True)`
  - e.g. `energy.plasma.plot.timeseries(other_run_energy)`
- `array.plasma.plot.lineout(*, x=None, ax=None, title=None, reference=None, x_of=None, xlabel=None, rationals=None, nfp=None, backend=None, **selection)`: Plot a one-dimensional profile after selecting every other dimension.
  - e.g. `phi.plasma.plot.lineout(x="eta1", t=-1, eta2=0.5, eta3=0)`
  - e.g. `T.plasma.plot.lineout(x="eta1", t=-1, reference=exact, x_of=lambda eta1: L * eta1)`
- `array.plasma.plot.line_animation(*, x=None, sweep='t', reference=None, x_of=None, xlabel=None, ylim=None, step=1, max_frames=None, interval=100, title=None, alongside=None, alongside_logy=False, backend=None, **selection)`: Animate a one-dimensional profile over sweep.
  - e.g. `T.plasma.plot.line_animation(reference={"exact": exact}, step=2)`
  - e.g. `u.plasma.plot.line_animation(x="eta1", alongside=[n, [en_U, en_B]], alongside_logy=True, eta2=0, eta3=0)`
- `array.plasma.plot.against_theory(theory=None, *, show_error=True, xlabel=None, ylabel=None, title=None, logx=False, logy=False, backend=None)`: Plot these measured values as points against a theory function.
  - e.g. `traced = spectrum.plasma.analysis.trace_branch(bohm_gross, k_range=(1.5, 5.5))`
  - e.g. `traced.omega.plasma.plot.against_theory(bohm_gross)`
- `array.plasma.plot.convergence(*others, order=None, xlabel=None, title='Convergence', ax=None, backend=None)`: Plot these errors against their resolution (or step size) on log-log axes, with the order.
  - e.g. `errors = xr.DataArray(l2_errors, dims="n", coords={"n": [16, 32, 64, 128]}, name="L2 error")`
  - e.g. `errors.plasma.plot.convergence(max_errors, backend="plotly")`
- `array.plasma.plot.vector(*, x, y, components=(0, 1), stride=1, coordinates='logical', ax=None, backend=None, **selection)`: Plot two vector components after selecting time and remaining dimensions.
  - e.g. `E.plasma.plot.vector(x="eta1", y="eta2", t=-1, eta3=0)`
- `array.plasma.plot.volume_slices(*, indices=None, cmap=None, backend=None, **selection)`: Render three orthogonal slices of a selected scalar volume.
  - e.g. `density.plasma.plot.volume_slices(t=-1)`
- `array.plasma.plot.volume(*, name=None, cmap='viridis', opacity='linear', **selection)`: Create a PyVista volume plotter for a selected scalar field.
  - e.g. `density.plasma.plot.volume(cmap="viridis", opacity="linear", t=-1).show()`
- `array.plasma.plot.isosurface(*, values=5, cmap='viridis', opacity=1.0, clim=None, show_domain=True, title=None, symmetric=False, robust=False, plotter=None, **selection)`: Draw PyVista contour surfaces of this scalar field in physical space.
  - e.g. `phi.plasma.plot.isosurface(values=[-0.5, 0.5], cmap="RdBu_r", t=0).show()`
- `array.plasma.plot.slices_3d(*, cuts=None, cmap='viridis', clim=None, show_domain=True, title=None, symmetric=False, robust=False, plotter=None, **selection)`: Draw PyVista surfaces of constant logical coordinate in physical space.
  - e.g. `phi.plasma.plot.slices_3d(cuts={"eta3": [0, 0.25, 0.5, 0.75]}, cmap="RdBu_r", t=0).show()`
- `array.plasma.plot.glyphs(*, components='cartesian', stride=2, scale=None, cmap='viridis', show_domain=True, title=None, plotter=None, **selection)`: Draw PyVista arrows of this (component, eta1, eta2, eta3) vector field.
  - e.g. `B.plasma.plot.glyphs(stride=3, t=-1).show()`
- `array.plasma.plot.streamlines(*, components='cartesian', n_points=100, source_radius=None, source_center=None, max_length=None, tube_radius=None, cmap='viridis', show_domain=True, title=None, plotter=None, **selection)`: Draw PyVista field lines of this vector field, e.g. magnetic field lines.
  - e.g. `B.plasma.plot.streamlines(n_points=60, source_center=(3.5, 0, 0), t=-1).show()`
- `array.plasma.plot.movie(path, *, kind='slices', step=1, framerate=10, clim=None, **options)`: Render one PyVista 3-D view per time step into a GIF or video.
  - e.g. `phi.plasma.plot.movie("mode.gif", kind="slices", cuts={"eta3": [0, 0.25, 0.5, 0.75]}, cmap="RdBu_r")`
- `array.plasma.plot.compare(other, *, mode='difference', ax=None, backend=None)`: Plot a one-dimensional aligned difference or ratio against another array.
  - e.g. `field.plasma.plot.compare(reference_field, mode="ratio")`
- `array.plasma.plot.overlay_orbits(orbits, *, x, y, max_markers=200, ax=None, cmap=None, backend=None, **selection)`: Plot this field's slice with marker orbit paths from orbits overlaid.
  - e.g. `phi.plasma.plot.overlay_orbits(orbits, x="eta1", y="eta2", t=-1, eta3=0)`
- `array.plasma.plot.dispersion(*, dim=None, detrend=True, branches=None, log=True, dynamic_range=6.0, kmin=None, kmax=None, omega_max=None, vmin=None, vmax=None, cmap=None, ax=None, title=None, frequencies=None, points=None, fits=(), backend=None)`: Plot the space-time power spectrum of this (t, dim) field as a dispersion relation.
  - e.g. `E.plasma.plot.dispersion(dim="eta1", branches={"Bohm-Gross": lambda k: np.sqrt(1 + 3 * k**2)})`
  - e.g. `spectrum = E.plasma.analysis.dispersion(dim="eta1")  # or pass the spectrum itself`
- `array.plasma.plot.power_spectrum(*, dims=None, detrend=True, window=None, peaks=None, band=None, frequencies=None, logy=True, omega_max=None, ax=None, title=None, backend=None, **selection)`: Plot the power per frequency bin, averaged over dims.
  - e.g. `band = phi.plasma.analysis.filter_time(dims=("eta1", "eta2", "eta3"))`
  - e.g. `phi.plasma.plot.power_spectrum(peaks=2, band=band, frequencies={"TAE gap": omega_tae})`
- `array.plasma.plot.filtered(result, *, ax=None, backend=None, **selection)`: Plot a probe of this signal against a filtered reconstruction.
  - e.g. `band = phi.plasma.analysis.filter_time(dims=("eta1", "eta2", "eta3"))`
  - e.g. `phi.plasma.plot.filtered(band, eta1=0.4, eta2=0.0, eta3=0.0)`
- `array.plasma.plot.spectrogram(*, length, step=None, detrend=True, window='hann', log=True, dynamic_range=4.0, omega_max=None, frequencies=None, ax=None, backend=None, **selection)`: Plot short-time power spectra over (t, omega).
  - e.g. `signal.plasma.plot.spectrogram(length=200.0, step=10.0, omega_max=0.45)`
- `array.plasma.plot.mode_amplitudes(*, dims=None, names=('m', 'n'), top=6, fit=None, reduce='max', scale=None, relative=False, logy=True, ax=None, backend=None, **selection)`: Plot the amplitude of the strongest (m, n) modes of this field over time.
  - e.g. `phi.plasma.plot.mode_amplitudes(top=2, fit=(100, 500))`
- `array.plasma.plot.mode_map(*, dims=None, m_range=None, n_range=None, reduce='max', scale=None, log=True, ax=None, backend=None, **selection)`: Plot |amplitude| over the (m, n) plane at one time.
  - e.g. `phi.plasma.plot.mode_map(t=-1, m_range=(0, 16))`
- `array.plasma.plot.radial_power(*, x=None, x_of=None, xlabel=None, continuum=None, detrend=True, window=None, log=True, dynamic_range=3.0, omega_max=None, ax=None, backend=None, **selection)`: Plot the time-power over (omega, x), averaged over the other dimensions.
  - e.g. `phi.plasma.plot.radial_power(x_of=lambda eta1: 0.1 + 0.9 * eta1, omega_max=0.5)`
- `array.plasma.plot.mode_profiles(omega=None, *, x=None, dims=None, x_of=None, xlabel=None, top=4, phase=True, scale=None, backend=None, **selection)`: Plot the radial profile of each (m, n) harmonic of this field.
  - e.g. `phi.plasma.plot.mode_profiles(omega, x_of=lambda eta1: 0.1 + 0.9 * eta1, top=2)`
  - e.g. `phi.plasma.plot.mode_profiles(t=-1, scale=(1, 6))`
- `array.plasma.plot.profiles(*, x=None, over='t', at=None, x_of=None, xlabel=None, ax=None, title=None, reference=None, backend=None, **selection)`: Plot profiles along x at several values of over in one axes.
  - e.g. `T.plasma.plot.profiles(x="eta1", at=[0, 10, 20, 40], reference={"exact": exact})`
- `array.plasma.plot.cross_spectrum(other, *, dims=None, detrend=True, window=None, omega_max=None, backend=None)`: Plot the magnitude, coherence and phase of other relative to this signal.
  - e.g. `u.plasma.plot.cross_spectrum(b, dims="eta3", omega_max=1.5)`
- `array.plasma.plot.pencil_fit(*, n_modes=1, pencil=None, detrend=False, backend=None, **selection)`: Plot a matrix-pencil fit of this (t,) series.
  - e.g. `probe.plasma.plot.pencil_fit(n_modes=1)`
- `array.plasma.plot.view(*, x=None, y=None, sweep='t', coords='logical', plane='XY', vmin=None, vmax=None, shared_clim=True, cmap=None, equal_aspect=None, title=None, symmetric=False, robust=False, levels=None, fill=True, overlays=None, xlabel=None, ylabel=None, colorbar_label=None, **selection)`: Configure a reusable slice view without rendering a figure.
  - e.g. `view = f.plasma.plot.view(x="eta1", y="v1", cmap="RdBu_r")`
  - e.g. `view.slice(t=-1)`
- `array.plasma.plot.slice(*, x=None, y=None, sweep='t', coords='logical', plane='XY', vmin=None, vmax=None, shared_clim=True, cmap=None, equal_aspect=None, title=None, symmetric=False, robust=False, levels=None, fill=True, overlays=None, xlabel=None, ylabel=None, colorbar_label=None, ax=None, backend=None, **selection)`: Render one 2-D slice.
  - e.g. `phi.plasma.plot.slice(x="eta1", y="eta2", t=-1)`
  - e.g. `n.plasma.plot.slice(coords="physical", plane="XY", t=-1, eta3=0, levels=[0.2])`
- `array.plasma.plot.panels(*, x=None, y=None, sweep='t', coords='logical', plane='XY', vmin=None, vmax=None, shared_clim=True, cmap=None, equal_aspect=None, title=None, symmetric=False, robust=False, levels=None, fill=True, overlays=None, xlabel=None, ylabel=None, colorbar_label=None, nrows=3, ncols=4, backend=None, **selection)`: Render evenly spaced snapshots along the sweep.
  - e.g. `phi.plasma.plot.panels(x="eta1", y="eta2", nrows=2, ncols=3, eta3=0)`
- `array.plasma.plot.viewer(*, x=None, y=None, sweep='t', coords='logical', plane='XY', vmin=None, vmax=None, shared_clim=True, cmap=None, equal_aspect=None, title=None, symmetric=False, robust=False, levels=None, fill=True, overlays=None, xlabel=None, ylabel=None, colorbar_label=None, backend=None, **selection)`: Create an interactive slider view; keep a reference to the returned viewer.
  - e.g. `viewer = phi.plasma.plot.viewer(x="eta1", y="eta2")`
- `array.plasma.plot.animation(*, x=None, y=None, sweep='t', coords='logical', plane='XY', vmin=None, vmax=None, shared_clim=True, cmap=None, equal_aspect=None, title=None, symmetric=False, robust=False, levels=None, fill=True, overlays=None, xlabel=None, ylabel=None, colorbar_label=None, interval=100, step=1, max_frames=None, alongside=None, backend=None, **selection)`: Animate the sweep; keep a reference to the returned Matplotlib animation.
  - e.g. `n.plasma.plot.animation(coords="physical", plane="XY", eta3=0, levels=[0.2])`
  - e.g. `vorticity.plasma.plot.animation(alongside=[density], eta3=0)`
- `array.plasma.plot.frames(directory, *, x=None, y=None, sweep='t', coords='logical', plane='XY', vmin=None, vmax=None, shared_clim=True, cmap=None, equal_aspect=None, title=None, symmetric=False, robust=False, levels=None, fill=True, overlays=None, xlabel=None, ylabel=None, colorbar_label=None, step=1, prefix='frame', dpi=110, **selection)`: Export the sweep as PNG frames.
  - e.g. `phi.plasma.plot.frames("frames", x="eta1", y="eta2", eta3=0, step=5)`
- `array.plasma.plot.trajectories(*, max_markers=200, show_paths=None, ax=None, backend=None)`: Plot the three-dimensional paths of saved markers, for an orbit product.

### array.plasma.plot.view(...)

A configured array view, shared by static, interactive and exported plots.

- `view.slice(*, ax=None, backend=None, **selection)`: Draw a snapshot, e.g. view.slice(t=-1).
  - e.g. `view.slice(t=-1)`
- `view.panels(*, nrows=3, ncols=4, backend=None)`: Draw snapshots spread evenly along the sweep.
  - e.g. `view.panels(nrows=2, ncols=3)`
- `view.viewer(*, backend=None)`: Create a viewer with sliders for the unselected dimensions.
- `view.animation(*, interval=100, step=1, max_frames=None, alongside=None, backend=None)`: Create a Matplotlib animation using this view's rendering options.
  - e.g. `view.animation(step=2)`
- `view.save_frames(directory, *, step=1, prefix='frame', dpi=110)`: Export PNG frames using this view's rendering options.
  - e.g. `view.save_frames("frames")`

### array.plasma.analysis

Quantitative diagnostics of one array, as array.plasma.analysis.<quantity>(...).

- `array.plasma.analysis.growth_rate(*, window=(None, None), amplitude=False)`: Fit exp(rate * t + intercept) to this time series within window.
  - e.g. `energy.plasma.analysis.growth_rate(window=(0.0, 5.0), amplitude=True).rate`
- `array.plasma.analysis.damping_rate(*, window=(None, None), amplitude=False)`: Fit exponential decay to the envelope of this oscillating time series.
  - e.g. `energy.plasma.analysis.damping_rate(amplitude=True).rate`
- `array.plasma.analysis.oscillation_frequency(*, window=(None, None), method='zero_crossings', detrend=True)`: Measure the frequency of this oscillating time series from its zero crossings or peaks.
  - e.g. `probe.plasma.analysis.oscillation_frequency(window=(5.0, 40.0)).omega`
- `array.plasma.analysis.map_coordinate(dim, mapping, *, name=None, units=None, label=None)`: Replace a coordinate by a function of it, e.g. eta1 by the minor radius in meters.
  - e.g. `r_T = T.plasma.analysis.map_coordinate("eta1", lambda eta1: 0.1 + 0.9 * eta1, name="r", units="m")`
  - e.g. `r_T.plasma.plot.profiles(x="r", eta2=0, eta3=0)`
- `array.plasma.analysis.envelope()`: Return the local maxima of this time series.
- `array.plasma.analysis.norm(*, dims=None, squared=False)`: Return the L2 norm over dims (default: every dimension except t).
  - e.g. `div_B.plasma.analysis.norm()`
- `array.plasma.analysis.drift(*, ref=None)`: Return the signed deviation of this time series from ref or from its first sample.
  - e.g. `energy.plasma.analysis.drift().plasma.plot.timeseries()`
- `array.plasma.analysis.relative_error(*, ref=None, skip_first=True)`: Return the absolute relative deviation from ref or from this series' first sample.
  - e.g. `energy.plasma.analysis.relative_error(ref=exact_solution)`
- `array.plasma.analysis.spatial_average(*, dims=None)`: Return the mean over the logical space dimensions eta1, eta2, eta3 (or dims).
  - e.g. `distribution.plasma.analysis.spatial_average()`
- `array.plasma.analysis.velocity_moments(*, dims=None)`: Return density, mean velocity and variance of a binned distribution over its velocity dimensions.
  - e.g. `distribution.plasma.analysis.velocity_moments()`
- `array.plasma.analysis.dispersion(*, dim=None, detrend=True)`: Return the space-time power spectrum of this (t, dim) field.
  - e.g. `spectrum = E.isel(eta2=0, eta3=0).plasma.analysis.dispersion()`
- `array.plasma.analysis.fit_branches(*, n_branches, k_range=None, noise_level=0.5, order=10)`: Fit straight dispersion branches (omega = v * k) to this (omega, k) power spectrum.
  - e.g. `field.plasma.analysis.dispersion().plasma.analysis.fit_branches(n_branches=2)`
- `array.plasma.analysis.fft(*, dim, detrend=False, window=None)`: Return the two-sided Fourier coefficients along dim.
  - e.g. `phi.plasma.analysis.fft(dim="eta1")`
- `array.plasma.analysis.time_fft(*, detrend=False, window=None)`: Return the one-sided temporal coefficients and the power per bin.
  - e.g. `phi.plasma.analysis.time_fft(detrend=True)`
- `array.plasma.analysis.filter_time(*, dims=None, omega_min=1e-08, pad_bins=0)`: Return the dominant frequency band, reconstructed.
  - e.g. `band = phi.plasma.analysis.filter_time(dims=("eta1", "eta2", "eta3"))`
- `array.plasma.analysis.band_filter(omega_lo, omega_hi, *, detrend=False)`: Return only the frequencies in [omega_lo, omega_hi].
  - e.g. `phi.plasma.analysis.band_filter(0.08, 0.11)`
- `array.plasma.analysis.spectral_peaks(*, n_peaks=3, dims=None, omega_min=1e-08, detrend=True, window=None)`: Return the strongest spectral peaks, with sub-bin frequencies.
  - e.g. `phi.plasma.analysis.spectral_peaks(n_peaks=2)`
- `array.plasma.analysis.spectrogram(*, length, step=None, detrend=True, window='hann')`: Return power spectra in sliding time windows.
  - e.g. `signal.plasma.analysis.spectrogram(length=200.0, step=10.0)`
- `array.plasma.analysis.mode_spectrum(*, dims=None, names=('m', 'n'), periods=None, scale=None)`: Return the complex amplitudes over poloidal/toroidal mode numbers.
  - e.g. `modes = phi.plasma.analysis.mode_spectrum()`
- `array.plasma.analysis.mode_amplitudes(*, top=None, real=True, relative=False)`: Return the real amplitudes of this mode spectrum along one mode dimension.
  - e.g. `phi.plasma.analysis.mode_spectrum().plasma.analysis.mode_amplitudes(top=4)`
- `array.plasma.analysis.mode_structure(omega, *, window='hann', detrend=True)`: Return the complex amplitude at the exact frequency omega at every point.
  - e.g. `phi.plasma.analysis.mode_structure(omega)`
- `array.plasma.analysis.cross_spectrum(other, *, dims=None, detrend=True, window=None)`: Return the cross-spectrum, phase (of other relative to this) and coherence.
  - e.g. `u.plasma.analysis.cross_spectrum(b, dims="eta3")`
- `array.plasma.analysis.matrix_pencil(*, n_modes=1, pencil=None, detrend=False)`: Return frequencies and growth rates beyond the FFT resolution.
  - e.g. `probe.plasma.analysis.matrix_pencil(n_modes=1)`
- `array.plasma.analysis.gradient(*, domain=None)`: Return the Cartesian gradient of this scalar field on a mapped domain.
  - e.g. `phi.plasma.analysis.gradient()`
- `array.plasma.analysis.surface_average(*, jacobian=None, domain=None, quadrature=None)`: Return the flux-surface average of this field over its two angles.
  - e.g. `ev.mod_B.plasma.analysis.surface_average(jacobian=ev.Jac)`
- `array.plasma.analysis.rational_surfaces(*, count=4, nfp=None, max_denominator=12)`: Return where this rotational transform (or safety factor) profile is a low-order rational.
  - e.g. `ev.iota.plasma.analysis.rational_surfaces(count=3)`
- `array.plasma.analysis.error(exact, *, norm='rms', relative=False, dims=None, weighted=False, domain=None, args=None)`: Return the error against an exact solution (an array or a function of the coordinates).
  - e.g. `T.plasma.analysis.error(exact, relative=True)`
  - e.g. `T.plasma.analysis.error(exact, norm="max")`
- `array.plasma.analysis.project_mode(*, dim, number, kind='sin', period=1.0, bin_correction=False)`: Return the amplitude of one Fourier mode along dim.
  - e.g. `rho.plasma.analysis.project_mode(dim="eta2", number=3, kind="complex")`
- `array.plasma.analysis.divergence(*, components='cartesian', domain=None)`: Return the divergence of this vector field.
  - e.g. `B.plasma.analysis.divergence()`
- `array.plasma.analysis.curl(*, components='cartesian', domain=None)`: Return the curl of this vector field, in Cartesian components.
  - e.g. `B.plasma.analysis.curl()`
- `array.plasma.analysis.flux_function()`: Return the flux (or stream) function of this 2-D in-plane field.
  - e.g. `B.plasma.analysis.flux_function()`
- `array.plasma.analysis.cylindrical_components()`: Return the Cartesian components rotated to (R, phi, Z).
  - e.g. `E.plasma.analysis.cylindrical_components()`
- `array.plasma.analysis.toroidal_components(*, R0, Z0=0.0)`: Return the Cartesian components rotated to (radial, poloidal, toroidal) about an axis at R0.
  - e.g. `u.plasma.analysis.toroidal_components(R0=3.0)`
- `array.plasma.analysis.polar_coordinates(*, center=(0.0, 0.0))`: Return this array with coordinates r and theta in the X-Y plane.
  - e.g. `n.plasma.analysis.polar_coordinates(center=(0.0, 0.0))`
- `array.plasma.analysis.trace_branch(theory, *, window=0.2, k_range=None, threshold=0.001)`: Return the measured frequency of a dispersion branch near theory(k) in this (omega, k) spectrum.
  - e.g. `spectrum.plasma.analysis.trace_branch(bohm_gross, window=0.2, k_range=(1.5, 5.5))`
- `array.plasma.analysis.drop_periodic_endpoint(dim, *, period=1.0)`: Return this array without a duplicated periodic endpoint along dim.

### array.plasma.data

The data behind each plot in ArrayPlots, without rendering it.

- `array.plasma.data.lineout(*, x=None, **selection)`: Return the 1-D profile ArrayPlots.lineout would plot.
  - e.g. `n.plasma.data.lineout(x="eta1", t=-1, eta2=0.3, eta3=0)`
- `array.plasma.data.vector(*, x, y, components=(0, 1), stride=1, coordinates='logical', **selection)`: Return the selected, strided vector field ArrayPlots.vector would plot.
- `array.plasma.data.volume_slices(*, indices=None, **selection)`: Return the three orthogonal planes ArrayPlots.volume_slices would plot.
- `array.plasma.data.grid(*, name=None, **selection)`: Return this field as a pyvista.StructuredGrid on its physical points.
  - e.g. `grid = phi.plasma.data.grid(t=-1)`
- `array.plasma.data.to_vtk(path, *, name=None, **selection)`: Write this field to VTK structured-grid files for ParaView.
  - e.g. `field.plasma.data.to_vtk("frames")`
- `array.plasma.data.slices_3d(*, cuts=None, **selection)`: Return the logical cuts ArrayPlots.slices_3d would draw.
- `array.plasma.data.compare(other, *, mode='difference')`: Return the aligned difference or ratio ArrayPlots.compare would plot.
  - e.g. `field.plasma.data.compare(reference_field, mode="ratio")`
- `array.plasma.data.view(*, x=None, y=None, sweep='t', coords='logical', plane='XY', **selection)`: Return every remaining dimension of this array, sweep included.
  - e.g. `n.plasma.data.view(coords="physical", plane="XY", eta3=0)`
- `array.plasma.data.slice(*, x=None, y=None, sweep='t', coords='logical', plane='XY', **selection)`: Return the single 2-D slice ArrayPlots.slice would plot.
  - e.g. `phi.plasma.data.slice(x="eta1", y="eta2", t=-1)`
  - e.g. `n.plasma.data.slice(coords="physical", plane="XY", t=-1, eta3=0)`
- `array.plasma.data.dispersion(*, dim=None, detrend=True)`: Return the space-time power spectrum ArrayPlots.dispersion would plot.
  - e.g. `field.plasma.data.dispersion(dim="eta1")`
- `array.plasma.data.overlay_orbits(orbits, *, x, y, max_markers=200, **selection)`: Return the field slice and marker subset ArrayPlots.overlay_orbits would plot.
  - e.g. `field, paths = field.plasma.data.overlay_orbits(orbits, x="eta1", y="eta2", t=-1)`
- `array.plasma.data.trajectories(*, max_markers=200)`: Return the marker-position subset ArrayPlots.trajectories would plot.
- `array.plasma.data.timeseries(*others)`: Return this time series and any others, validated, as ArrayPlots.timeseries plots them.
  - e.g. `energy.plasma.data.timeseries(other_run_energy)`

### dataset.plasma.plot

Plots of one dataset, as dataset.plasma.plot.<kind>(...).

- `dataset.plasma.plot.power_spectrum(*, backend=None, **options)`: Plot the power of a time_fft Dataset.
  - e.g. `phi.plasma.analysis.time_fft(detrend=True).plasma.plot.power_spectrum(peaks=2)`
- `dataset.plasma.plot.cross_spectrum(*, omega_max=None, backend=None)`: Plot the magnitude, coherence and phase of a cross_spectrum Dataset.
  - e.g. `u.plasma.analysis.cross_spectrum(b, dims="eta3").plasma.plot.cross_spectrum(omega_max=1.5)`
- `dataset.plasma.plot.trajectories(*, max_markers=200, show_paths=None, ax=None, backend=None)`: Plot the three-dimensional paths of saved markers, for an orbits product.
  - e.g. `orbits.plasma.plot.trajectories(max_markers=200)`
- `dataset.plasma.plot.scatter(*, x, y, color=None, ax=None, cmap=None, s=8, color_at=None, background=None, background_options=None, backend=None, **selection)`: Scatter two position variables, optionally colored by a third (e.g. density or a tracer).
  - e.g. `markers.plasma.plot.scatter(x="x", y="y", color="density", t=-1)`
  - e.g. `markers.plasma.plot.scatter(x="x", y="y", color="tracer", color_at=0, t=-1)`
- `dataset.plasma.plot.orbit_classification(*, x='v_par', y=None, v_par='v_par', t=0, ax=None, s=8, backend=None)`: Plot markers in a phase-space plane, colored as passing, trapped or lost.
  - e.g. `orbits.plasma.plot.orbit_classification()`
  - e.g. `orbits.plasma.plot.orbit_classification(x="p_phi")`
- `dataset.plasma.plot.animation(*, x, y, color=None, color_at=None, background=None, background_options=None, step=1, max_frames=None, interval=100, s=8, cmap=None, trail=None, paths=False, backend=None)`: Animate the markers moving over time, optionally over a field animated in sync.
  - e.g. `markers.plasma.plot.animation(x="x", y="y", color="density", background=n, step=2)`
  - e.g. `orbits.plasma.plot.animation(x="R", y="z", color="classification", trail=300, paths=True, background=psi)`
- `dataset.plasma.plot.paths(*, x='x', y='y', markers=6, near=None, background=None, background_options=None, t=0, ax=None, backend=None)`: Plot the paths of a few markers in a plane, with start and end markers.
  - e.g. `orbits.plasma.plot.paths(markers=4, background=psi.isel(t=0))`
- `dataset.plasma.plot.poloidal(*, color_by='classification', max_markers=200, boundary=None, ax=None, backend=None)`: Plot orbits projected onto the poloidal plane (R against z).
  - e.g. `orbits.plasma.plot.poloidal(boundary=field)`
- `dataset.plasma.plot.orbit_grid(*, markers=8, ncols=4, boundary=None, backend=None)`: Plot one small poloidal panel per marker, colored by orbit class.
  - e.g. `orbits.plasma.plot.orbit_grid(markers=8, ncols=4, boundary=field)`
  - e.g. `orbits.plasma.plot.orbit_grid(markers=[3, 17, 42])`
- `dataset.plasma.plot.quantities(*, quantities=('v_par', 'mu'), markers=6, drift_of=('mu',), backend=None)`: Plot saved orbit quantities over time for a few markers.
  - e.g. `orbits.plasma.plot.quantities(markers=4)`
- `dataset.plasma.plot.orbits_3d(*, color_by='t', max_markers=200, tube_radius=None, cmap=None, domain=None, title=None, plotter=None)`: Draw PyVista 3-D orbit lines, colored by "t", "classification" or any variable.
  - e.g. `orbits.plasma.plot.orbits_3d(color_by="classification", domain=phi.isel(t=0)).show()`

### dataset.plasma.analysis

Quantitative diagnostics of one dataset, as dataset.plasma.analysis.<quantity>(...).

- `dataset.plasma.analysis.classify_orbits(*, v_par='v_par')`: Classify each marker of this guiding-center orbits product: passing (0), trapped (1) or lost (-1).
  - e.g. `orbits.plasma.analysis.classify_orbits()`
- `dataset.plasma.analysis.orbit_invariants(*, absB=None)`: Return the speed, guiding-centre energy and pitch of the saved orbits.
  - e.g. `orbits.plasma.analysis.orbit_invariants(absB=absB_xyz)`
- `dataset.plasma.analysis.bounce_period(*, v_par='v_par')`: Return the bounce period of each trapped marker.
  - e.g. `orbits.plasma.analysis.bounce_period()`
- `dataset.plasma.analysis.surface_average(name, *, jacobian='Jac', domain=None, quadrature=None)`: Return the flux-surface average of one variable, with this Dataset's Jacobian.
  - e.g. `ev.plasma.analysis.surface_average("mod_B")`

### dataset.plasma.data

The data behind each plot in DatasetPlots, without rendering it.

- `dataset.plasma.data.trajectories(*, max_markers=200)`: Return the marker-position subset DatasetPlots.trajectories would plot.
  - e.g. `markers.plasma.data.trajectories(max_markers=50)`
- `dataset.plasma.data.scatter(*, x, y, color=None, color_at=None, **selection)`: Return the per-marker positions and colors DatasetPlots.scatter would plot.
  - e.g. `markers.plasma.data.scatter(x="x", y="y", color="density", t=-1).to_dataframe()`
  - e.g. `markers.plasma.data.scatter(x="x", y="y", color="x", color_at=0, t=-1)  # colored by the start`
- `dataset.plasma.data.orbit_classification(*, x='v_par', y=None, v_par='v_par', t=0)`: Return the per-marker x, y and classification that orbit_classification plots.
  - e.g. `orbits.plasma.data.orbit_classification(x="p_phi")`

### out.plot (a Struphy Output)

Plots of a whole run, as out.plot.<kind>(...), constructed as OutputPlots(out).

- `out.plot.scalars(names=None, *, relative_to=None, logy=False, backend=None)`: Overview of the scalar time series in one axes.
  - e.g. `out.plot.scalars()`
  - e.g. `out.plot.scalars(["en_U", "en_B"], logy=True)`
- `out.plot.energies(*, parts=None, total='en_tot', groups=None, logy=False, backend=None)`: Plot the run's energy budget from its energy scalars (en_* or *_energy).
  - e.g. `out.plot.energies()`
  - e.g. `out.plot.energies(groups={"wave": ["en_U", "en_B", "en_p"], "energetic ions": ["en_fv", "en_fB"]})`
- `out.plot.equilibrium(ax=None, *, backend=None)`: Plot radial profiles of this run's fluid equilibrium (out.equil, out.domain).
  - e.g. `out.plot.equilibrium()`
- `out.plot.equilibrium_3d(*, scalars='p0', cmap='viridis')`: Create a PyVista view of this run's fluid equilibrium; call .show() on the returned plotter.
  - e.g. `out.plot.equilibrium_3d(scalars="p0").show()`
- `out.plot.domain_3d(*, n1=8, n2=32, n3=32, surface=True)`: Draw a PyVista wireframe of this run's mapping (out.domain); call .show() on it.
  - e.g. `out.plot.domain_3d().show()`
  - e.g. `out.plot.domain_3d(n3=1).show()`
- `out.plot.profile`: Plots of this run's timing regions, e.g. out.plot.profile.gantt().
- `out.plot(*args, **kwargs)`: Show the quick default plot: an overview of every scalar time series.
  - e.g. `out.plot()`
  - e.g. `out.plot(logy=True)`

### out.plot.profile

Plots of one run's timing regions (out.profile.results), as out.plot.profile.<kind>(...).

- `out.plot.profile.gantt(*, return_fig=True, verbose=False, **kwargs)`: A timeline of every recorded region, one row per rank.
  - e.g. `out.plot.profile.gantt()`
- `out.plot.profile.flame(*, return_fig=True, verbose=False, **kwargs)`: A flame chart reconstructing the call stack from region timings.
  - e.g. `out.plot.profile.flame()`
- `out.plot.profile.callgraph(*, return_fig=True, verbose=False, **kwargs)`: The explicit call graph (which region calls which), without timings.
  - e.g. `out.plot.profile.callgraph()`

### out.analysis (a Struphy Output)

Spectral diagnostics of a run's products, as out.analysis.<kind>(product, ...).

- `out.analysis.fft(product, *, dim, detrend=False, window=None)`: Compute two-sided Fourier coefficients of a product along dim.
  - e.g. `out.analysis.fft(phi.isel(t=-1, eta2=0, eta3=0), dim="eta1")`
- `out.analysis.time_fft(product, *, detrend=False, window=None)`: Compute one-sided temporal Fourier coefficients and per-bin power of a product.
  - e.g. `out.analysis.time_fft(phi.isel(eta1=0.5, eta2=0, eta3=0), window="hann")`
- `out.analysis.filter_time(product, *, dims=None, omega_min=1e-08, pad_bins=0)`: Reconstruct the dominant temporal frequency band of a product.
  - e.g. `result = out.analysis.filter_time(phi)`
  - e.g. `result.filtered.plasma.plot.slice(t=-1, eta3=0)`
- `out.analysis.linear_mhd_energies(*, velocity='mhd/velocity', b_field='em_fields/b_field', pressure='mhd/pressure', gamma=1.6666666666666667)`: Compute LinearMHD's energy scalars from fields, as a Dataset of time series.
  - e.g. `energies = out.analysis.linear_mhd_energies()`
  - e.g. `filtered = out.analysis.filter_time(out.evaluate("mhd/velocity", representation="2")).filtered`
- `out.analysis.quadrature_grid()`: Return the run's Gauss-Legendre quadrature points and weights in each logical direction.
  - e.g. `etas, weights = out.analysis.quadrature_grid()`
  - e.g. `b = out.evaluate("em_fields/b_field", eta1=etas["eta1"], eta2=etas["eta2"], eta3=etas["eta3"], representation="2")`
- `out.analysis.mode_spectrum(product, *, dims=('eta2', 'eta3'), names=('m', 'n'), periods=1.0)`: Compute complex amplitudes of a product over poloidal/toroidal mode numbers.
  - e.g. `out.analysis.mode_spectrum("em_fields/phi_log")`
  - e.g. `out.analysis.mode_spectrum(phi.isel(t=-1), dims="eta3", names="n")`

## Top level

- `plasma_plots.figure(nrows=1, ncols=1, *, backend=None, sharex=False, sharey=False, figsize=None, title=None, **options)`: Compose several plots into one figure, drawn with Matplotlib or as one Plotly figure.
  - e.g. `with plasma_plots.figure(2, 1, sharex=True, backend="plotly") as fig: energy.plasma.plot.timeseries(fit=(0.0, 5.0), ax=fig[0]) drift.plasma.plot.timeseries(logy=True, ax=fig[1])`
  - e.g. `fig.save("energies.html")`
- `plasma_plots.set_backend(backend)`: Set the backend of every plot that does not pass backend= itself.
  - e.g. `previous = plasma_plots.set_backend("plotly")`
  - e.g. `phi.plasma.plot.slice(t=-1, eta3=0)  # a Plotly figure`
- `plasma_plots.get_backend()`: Return the backend of plots that do not pass backend= themselves.
- `plasma_plots.mpi_rank()`: Return this process' rank in MPI_COMM_WORLD, without initializing MPI.
  - e.g. `mpi_rank()  # in a serial run`
- `plasma_plots.is_plotting_rank()`: Tell whether this process draws plots and writes their files.
- `plasma_plots.plotting.PlotResult.save(path, *, close=False, frame=None, **kwargs)`: Save the figure to a file, as drawn.
- `plasma_plots.plotting.PlotResult.show()`: Show the figure with matplotlib.pyplot.show, or a Plotly figure with its show.
- `plasma_plots.plotting.PlotResult.to_plotly(*, close=False)`: The same result with the figure converted to an interactive Plotly figure.
  - e.g. `plot_timeseries(energy, fit=GrowthFit(window=(5.0, 20.0))).to_plotly().save("energy.html")`
- `plasma_plots.plotly_backend.to_plotly(figure, *, strict=False)`: Convert a drawn Matplotlib figure into an interactive Plotly figure.
  - e.g. `result = plot_slice(phi.isel(t=-1, eta3=0))`
  - e.g. `to_plotly(result.fig).write_html("phi.html")`
- `plasma_plots.plotly_backend.animation_to_plotly(animation, *, labels=None, prefix=None, play=True, strict=False)`: Convert a Matplotlib animation of the plotting functions into a Plotly figure with frames.
  - e.g. `animation = animate_slices(phi.isel(eta3=0), step=2)`
  - e.g. `animation_to_plotly(animation).write_html("phi.html")`

## Result types

What the methods return, with their fields (`help()` on the class explains each).

- `PlotResult(fig, ax, artists, fit_results, data)`: An already-drawn figure and what was drawn; saving never redraws it.
- `FitResult(rate, intercept, time, fitted)`: An exponential fit exp(rate t + intercept), from growth_rate or damping_rate.
- `OscillationFit(omega, period, times, method)`: The frequency of an oscillating time series, from oscillation_frequency.
- `ConvergenceFit(order, constant, sizes, fitted)`: A power law error = constant * size**order, from convergence_order.
- `BranchFit(velocity, k, omega)`: A dispersion branch fitted as omega = velocity * k, from fit_dispersion_branches.
- `TimeFilterResult(filtered, spectrum)`: The result of filter_time: filtered field and reduced spectrum with the band.
- `GrowthFit(window, amplitude_from_quadratic)`: Configuration for an exponential growth-rate fit.
- `View(x, y, sweep, select, isel, coordinates, plane)`: A reusable selection and rendering recipe for an N-dimensional product.

## Plain functions behind the accessors

The same plots and diagnostics as functions of arrays, e.g. `plot_slice(phi.isel(t=-1))`; each accessor method names its function first under See Also, and `help()` shows the parameters.

- `plasma_plots.plotting`: `save_figure`, `shared_run_label`, `logical_grids`, `physical_grids`, `prepare_view`, `plot_timeseries`, `prepare_lineout`, `plot_lineout`, `prepare_vector`, `plot_vector`, `prepare_volume_slices`, `plot_volume_slices`, `prepare_compare`, `plot_compare`, `pyvista_volume`, `show_equilibrium`, `color_limits`, `plot_slice`, `plot_panels`, `animate_slices`, `animate_fields`, `save_frames`, `plot_scalars`, `plot_convergence`, `plot_dispersion`, `save_all_scalars`, `prepare_orbits`, `plot_marker_trajectories`, `resolve_marker_selection`, `prepare_marker_scatter`, `plot_marker_scatter`, `animate_markers`, `plot_marker_paths`, `plot_field_with_orbits`, `prepare_orbit_classification`, `plot_orbit_classification`, `prepare_continuous_spectrum`, `plot_continuous_spectrum`, `plot_equilibrium_profile`, `energy_names`, `plot_energy_budget`, `plot_profiles`, `plot_orbit_poloidal`, `plot_orbit_quantities`, `animate_lines`, `plot_measured_vs_theory`, `plot_orbit_grid`
- `plasma_plots.spectral_plots`: `plot_power_spectrum`, `plot_filtered`, `plot_spectrogram`, `plot_mode_amplitudes`, `plot_mode_map`, `plot_radial_power`, `plot_mode_profiles`, `plot_cross_spectrum`, `plot_pencil_fit`
- `plasma_plots.analysis`: `convergence_order`, `growth_rate`, `envelope`, `damping_rate`, `oscillation_frequency`, `norm`, `drift`, `spatial_average`, `velocity_moments`, `relative_error`, `classify_orbits`, `fit_dispersion_branches`, `power_spectrum`, `quadrature_weights`, `volume_integral`, `surface_average`, `rational_surfaces`, `field_energy`, `gradient`, `evaluate_on`, `error`, `project_mode`, `divergence`, `curl`, `flux_function`, `cylindrical_components`, `toroidal_components`, `polar_coordinates`, `orbit_invariants`, `bounce_period`
- `plasma_plots.spectral`: `hann`, `fft`, `time_fft`, `inverse_time_fft`, `fwhm_window`, `filter_time`, `drop_periodic_endpoint`, `band_filter`, `spectral_peaks`, `spectrogram`, `mode_spectrum`, `mode_amplitudes`, `mode_structure`, `cross_spectrum`, `matrix_pencil`, `pencil_reconstruction`, `trace_branch`
- `plasma_plots.arrays`: `logical_dims`, `angle_period`, `validate_array`, `axis_label`, `map_coordinate`, `value_label`, `scalar_names`, `save_scalars`, `periodicity`, `close_periodic`, `logical_derivative`, `mapping_jacobian`
- `plasma_plots.gvec`: `is_gvec`, `from_gvec`
- `plasma_plots.desc`: `from_desc`
- `plasma_plots.pyvista_plots`: `is_flat`, `structured_grid`, `push_forward`, `boundary_keys`, `boundary_faces`, `pyvista_isosurface`, `prepare_slices_3d`, `pyvista_slices`, `pyvista_glyphs`, `pyvista_streamlines`, `orbit_polylines`, `pyvista_orbits`, `pyvista_domain`, `save_vtk`, `save_movie`

## Command line: plasma-plots

`plasma-plots info PATH`, `plot PATH PRODUCT METHOD key=value ... -o FILE` (any plot method above), `movie PATH PRODUCT ... -o FILE` and `quicklook PATH -o DIR`, on a Struphy run folder or a file xarray reads; `key=value` values: `t=-1` a position, `t=0.35` a value, `other=@name` another product. `plasma-plots COMMAND --help` for details. The same from Python, in `plasma_plots.cli`:

- `open_source(path, *, pproc=False)`: Open path: a Struphy Output for a run folder, else an xarray.Dataset.
- `parse_value(text, source=None)`: A key=value value as Python: an int, a float, a bool, None, a list or a string.
- `plot_methods(obj)`: The plot methods of obj that plasma-plots plot can call and save.
- `quicklook_plot(array)`: The plot method and options of a quick look at array, or None if there is none.
- `quicklook(source, directory, *, formats=('png',), dpi=None, log=None)`: Save the standard figures of source into directory; return the files written.
- `main(argv=None)`: Run the plasma-plots command.

## Theory: plasma_plots.theory

Analytic results to compare with, plain numpy; complex ω for dispersion relations.

### plasma_plots.theory.kinetic

- `susceptibility(omega, k, species, derivative=0)`: Compute the electrostatic susceptibility χ_s(ω, k) of one Maxwellian species.
- `electrostatic_dielectric(omega, k, species=None, derivative=0)`: Compute the electrostatic dielectric function ε(ω, k) = 1 + Σ_s χ_s of Maxwellian species.
- `solve_dispersion(function, k, guess, derivative=None, continuation=True, tol=1e-11, maxiter=60)`: Find a complex root ω(k) of a dispersion relation for every wavenumber.
- `bohm_gross(k)`: Compute the Bohm–Gross frequency ω = √(1 + 3k²) of Langmuir waves.
- `landau_damping_weak(k)`: Compute the weak-damping approximation of Langmuir waves: Bohm–Gross plus Landau damping.
- `langmuir(k)`: Compute the exact kinetic Langmuir root: the least-damped zero of ε(ω, k) near Bohm–Gross.
- `ion_acoustic_fluid(k, temperature_ratio=10.0, mass_ratio=1836.15267343, adiabatic_index=3.0)`: Compute the fluid ion-acoustic frequency with Boltzmann electrons and adiabatic ions.
- `ion_acoustic(k, temperature_ratio=10.0, mass_ratio=1836.15267343)`: Compute the kinetic ion-acoustic root of Maxwellian electrons and ions.
- `two_stream_cold(k, beam_speed, beam_density=0.5, all_roots=False)`: Compute the cold symmetric two-stream frequencies in closed form.
- `beam_plasma_cold(k, beam_speed, beam_density=0.1, plasma_density=1.0, all_roots=False)`: Compute the cold beam-plasma frequencies: the roots of a quartic.
- `two_stream(k, beam_speed, thermal_speed, beam_density=0.5)`: Compute the kinetic purely growing root of two warm counter-streaming electron beams.
- `bump_on_tail(k, beam_density=0.1, beam_speed=4.5, beam_thermal_speed=0.5, bulk_density=None)`: Compute the most unstable kinetic root of a bump-on-tail distribution.
- `maximum_growth(function, k_range, samples=64, tol=1e-08)`: Find the wavenumber of maximum growth rate of a dispersion relation.
- `weibel(k, anisotropy, parallel_thermal_speed)`: Compute the purely growing (or damped) root of the electron Weibel instability.

### plasma_plots.theory.waves

- `light_wave(k, c=1.0)`: Compute the frequency ω = c|k| of a light wave in vacuum.
- `plasma_light_wave(k, plasma_frequency=1.0, c=1.0)`: Compute the frequency of a light wave in an unmagnetized cold plasma, ω² = ω_p² + c²k².
- `magnetosonic_speeds(theta=0.0, alfven_speed=1.0, sound_speed=0.5)`: Compute the phase speeds of the three ideal-MHD waves (the Friedrichs diagram).
- `mhd_waves(k, theta=0.0, alfven_speed=1.0, sound_speed=0.5)`: Compute the frequencies of the shear Alfvén, slow and fast waves of ideal MHD.
- `dissipative_alfven(k, alfven_speed=1.0, resistivity=0.0, viscosity=0.0, theta=0.0)`: Compute the frequencies of shear Alfvén waves with resistivity and viscosity.
- `hall_mhd_parallel(k, alfven_speed=1.0, ion_inertial_length=1.0)`: Compute the whistler and ion-cyclotron branches of Hall MHD along B₀.
- `electron_ion(plasma_frequency=1.0, cyclotron_frequency=1.0, mass_ratio=1836.15267343, charge=1)`: Build the species of a quasi-neutral electron–ion plasma from the electron frequencies.
- `stix(omega, species)`: Compute Stix's cold-plasma dielectric parameters S, D, P, R and L.
- `refractive_index(omega, theta, species)`: Compute the two squared refractive indices n² = c²k²/ω² of cold-plasma waves.
- `appleton_hartree(omega, theta, plasma_frequency=1.0, cyclotron_frequency=1.0)`: Compute the Appleton–Hartree refractive indices of a cold magnetized electron plasma.
- `cold_plasma_waves(k, theta, species, c=1.0)`: Compute all positive-frequency branches ω(k) of the cold magnetized plasma.
- `cutoffs(species)`: Compute the cold-plasma cutoffs: the positive frequencies where R, L or P vanish.
- `resonances(theta, species)`: Compute the cold-plasma resonances at the angle θ: the positive frequencies where n² → ∞.
- `faraday_rotation(omega, length, species, c=1.0)`: Compute the Faraday rotation angle of a linearly polarized wave traveling along B₀.
- `group_velocity(omega_of_k, k, step=None)`: Compute the group velocity dω/dk of a dispersion relation by central differences.
- `cavity_modes(lengths, c=1.0, max_index=6, max_frequency=None)`: List the resonant modes of a rectangular cavity with perfectly conducting walls.
- `drift_wave(ky, kx=0.0, diamagnetic_speed=1.0, rho_s=1.0)`: Compute the frequency of the electron drift wave with adiabatic electrons.
- `hasegawa_wakatani(ky, kx=0.0, adiabaticity=1.0, gradient=1.0, viscosity=0.0)`: Compute the two linear modes of the Hasegawa–Wakatani equations.
- `parallel_wavenumber(r, m, n, q, major_radius=1.0)`: Compute the parallel wavenumber k∥ = (n + m/q(r))/R₀ of a Fourier mode in a cylinder.
- `alfven_continuum(r, m, n, q, major_radius=1.0, alfven_speed=1.0)`: Compute the shear Alfvén continuum ω(r) = |k∥(r)| v_A(r) of a cylinder.
- `slow_continuum(r, m, n, q, major_radius=1.0, alfven_speed=1.0, sound_speed=0.5)`: Compute the slow (cusp) continuum ω(r) = |k∥| c_s v_A/√(c_s² + v_A²) of a cylinder.
- `tae_frequency(q, major_radius=1.0, alfven_speed=1.0)`: Compute the frequency ω_TAE = v_A/(2|q|R₀) at the center of the toroidal Alfvén gap.

### plasma_plots.theory.parameters

- `plasma_frequency(density, mass=9.1093837015e-31, charge=1.602176634e-19)`: Compute the plasma frequency ω_p = √(n q² / (ε₀ m)).
- `cyclotron_frequency(field, mass=9.1093837015e-31, charge=1.602176634e-19)`: Compute the cyclotron (gyro-) frequency Ω = q|B|/m.
- `thermal_speed(temperature, mass=9.1093837015e-31, convention='sqrt(T/m)')`: Compute the thermal speed of a Maxwellian of temperature T.
- `debye_length(density, temperature)`: Compute the electron Debye length λ_D = √(ε₀ T / (n e²)).
- `larmor_radius(field, temperature=None, perpendicular_speed=None, mass=9.1093837015e-31, charge=1.602176634e-19)`: Compute the Larmor (gyro-) radius ρ = m v⊥ / (|q| B).
- `inertial_length(density, mass=9.1093837015e-31, charge=1.602176634e-19)`: Compute the inertial length (skin depth) d = c / ω_p.
- `alfven_speed(field, density, mass_number=1, relativistic=False)`: Compute the Alfvén speed v_A = B / √(μ₀ n m_i), with m_i = mass_number × m_p.
- `sound_speed(electron_temperature, ion_temperature=0.0, mass_number=1, charge_number=1, electron_gamma=1.0, ion_gamma=3.0)`: Compute the ion sound speed c_s = √((γ_e Z T_e + γ_i T_i) / m_i).
- `plasma_beta(density, temperature, field)`: Compute the plasma beta β = n T / (B²/(2μ₀)), the ratio of thermal to magnetic pressure.
- `plasma_parameter(density, temperature)`: Compute the plasma parameter N_D = (4π/3) n λ_D³, the number of electrons in a Debye sphere.
- `lower_hybrid_frequency(density, field, mass_number=1, charge_number=1)`: Compute the lower hybrid frequency, 1/ω_LH² = 1/(Ω_i² + ω_pi²) + 1/(|Ω_e| Ω_i).
- `upper_hybrid_frequency(density, field)`: Compute the upper hybrid frequency ω_UH = √(ω_pe² + Ω_e²).
- `struphy_units(x=1.0, B=1.0, n=1.0, kBT=None, velocity_scale='light', mass_number=None, charge_number=None)`: Compute the units of Struphy's normalization from its base units.
- `struphy_equation_parameters(units, charge_number=1, mass_number=1)`: Compute the equation parameters α, ε and κ of one Struphy species.

### plasma_plots.theory.orbits

- `gyrofrequency(field, charge=1.0, mass=1.0)`: Compute the gyrofrequency Ω = q|B|/m.
- `gyroradius(perpendicular_speed, field, charge=1.0, mass=1.0)`: Compute the gyroradius ρ = m v⊥ / (|q| B).
- `magnetic_moment(perpendicular_speed, field, mass=1.0)`: Compute the magnetic moment μ = m v⊥² / (2B), the adiabatic invariant of the gyromotion.
- `exb_drift(E, B)`: Compute the E×B drift velocity v_E = E × B / B².
- `grad_b_drift(perpendicular_speed, B, grad_B, charge=1.0, mass=1.0)`: Compute the grad-B drift velocity v_∇B = (m v⊥² / (2q)) B × ∇B / B³.
- `trapped_fraction(epsilon, approximation='exact')`: Compute the effective fraction of trapped particles on a flux surface of a circular tokamak.
- `pitch_parameter(pitch, epsilon, theta=0.0)`: Compute the pitch parameter λ = μB₀/E from the pitch v∥/v at the poloidal angle θ.
- `trapping_boundary(epsilon, quantity='lambda', theta=0.0)`: Compute the trapped–passing boundary on a flux surface.
- `trapping_parameter(lam, epsilon)`: Compute the trapping parameter κ² = (1 + ε − λ) / (2ε).
- `bounce_frequency(speed, kappa2, epsilon, safety_factor, major_radius)`: Compute the bounce frequency of a trapped particle, ω_b = π v √(2ε) / (4 q R₀ K(κ²)).
- `transit_frequency(speed, kappa2, epsilon, safety_factor, major_radius)`: Compute the poloidal transit frequency of a passing particle.
- `banana_width(gyroradius, kappa2, epsilon, safety_factor)`: Compute the full radial width of a banana orbit, Δr = 2√2 q κ ρ / √ε.

### plasma_plots.theory.exact

- `star_state(left, right, gamma=1.4)`: Solve for the star region of the Riemann problem of the 1-D Euler equations.
- `riemann_euler(x, t, left, right, gamma=1.4, x0=0.0)`: Compute the exact solution of the Riemann problem of the 1-D Euler equations.
- `sod_shock_tube(x, t, gamma=1.4, x0=0.5)`: Compute the exact solution of Sod's shock tube.
- `dam_break(x, t, depth=1.0, gravity=1.0, x0=0.0)`: Compute Ritter's exact dam-break solution of the 1-D shallow-water equations on a dry bed.
- `heat_kernel(x, t, diffusivity, width=0.0, center=0.0, mass=1.0)`: Compute a spreading Gaussian, the solution of the diffusion equation ∂u/∂t = D ∇²u.
- `advected(profile, x, t, velocity, period=None)`: Compute a profile carried unchanged at constant velocity, u(x, t) = u0(x − v t).
- `dalembert(x, t, initial, speed, initial_rate=None)`: Compute d'Alembert's solution of the 1-D wave equation ∂²u/∂t² = c² ∂²u/∂x².
- `pressureless(q, t, velocity, density=None, velocity_derivative=None)`: Follow a 1-D pressureless flow (the Zel'dovich approximation) in Lagrangian coordinates.
- `pressureless_eulerian(x, t, velocity, density=None, velocity_derivative=None)`: Compute a 1-D pressureless flow as a function of the position, before shell crossing.
- `caustic_time(velocity, q)`: Compute the time of the first caustic (shell crossing) of a pressureless flow.

### plasma_plots.theory.numerics

- `amplification_factor(omega_dt, method)`: Compute the amplification factor of a time integrator on the oscillator y' = iωy.
- `amplitude_error(omega_dt, method)`: Compute the relative amplitude error per step, |G| − 1, of a time integrator.
- `phase_error(omega_dt, method)`: Compute the relative frequency error of the numerical oscillation of a time integrator.
- `stability_limit(method)`: Return the largest stable |ω dt| of a time integrator on the imaginary axis.
- `numerical_frequency(omega, dt, method)`: Compute the complex frequency that a time integrator produces for a real frequency ω.
- `spline_galerkin_dispersion(k, dx, degree, c=1.0)`: Compute the numerical frequency of the periodic B-spline Galerkin wave equation in 1-D.

### plasma_plots.theory.special

- `faddeeva(z)`: Compute the Faddeeva function w(z) = exp(−z²) erfc(−iz).
- `plasma_dispersion(zeta, derivative=0)`: Compute the plasma dispersion function Z(ζ) of Fried and Conte, or its first derivative.
- `elliptic_k(m)`: Compute the complete elliptic integral of the first kind K(m), with parameter m = k².
- `elliptic_e(m)`: Compute the complete elliptic integral of the second kind E(m), with parameter m = k².
