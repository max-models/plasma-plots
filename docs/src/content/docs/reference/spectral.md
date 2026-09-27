---
title: struphy_plots.spectral
description: API reference for the spectral diagnostics and their plots.
---

Fourier and spectral diagnostics in `struphy_plots.spectral`, their plots in
`struphy_plots.spectral_plots`, and the matching accessor methods. See the
[Spectral analysis guide](/struphy-plots/guides/spectral/) for the conventions
(forward normalization by `N`, angular frequencies, numpy's sign) and for
figures.

## Transforms

### `fft(data, *, dim, detrend=False, window=None)`
Two-sided, shifted FFT along a uniform coordinate, divided by `N`. `dim` is
replaced by `omega` for `t`, otherwise by `k_<dim>`. It accepts complex data.
`window` is `None` or `"hann"`. Also `array.struphy.analysis.fft(dim=...)`.

### `time_fft(data, *, detrend=False, window=None)`
One-sided time FFT of a real signal. Returns a Dataset with complex
`coefficients` and `power` (mean square per bin; positive frequencies doubled
except the zero frequency and the even-`N` Nyquist bin). Coordinates that depend
on `t` are dropped. Also `array.struphy.analysis.time_fft()`.

### `inverse_time_fft(coefficients, template)`
Inverts `time_fft` coefficients, taking the length and coordinates from
`template`. A window or mean subtraction is not undone.

### `drop_periodic_endpoint(data, dim, *, period=1.0)`
Drops the last sample along `dim` if it repeats the first one period later.

## Filtering

### `filter_time(data, *, dims=None, omega_min=1e-8, pad_bins=0)`
Keeps the dominant peak's half-power band, chosen from the power summed over
`dims` (default: all but `t` and `component`), and rebuilds the signal.
Returns a `TimeFilterResult` with `.filtered` and `.spectrum` (`power`,
`dominant_frequency`, `idx_dominant`, `idx_lo`/`idx_hi`, `omega_lo`/`omega_hi`,
`has_peak`). A signal with no oscillation gives `has_peak=False`, NaN
frequencies and a zero reconstruction.

### `band_filter(data, omega_lo, omega_hi, *, detrend=False)`
Keeps only the frequencies in `[omega_lo, omega_hi]`.

### `fwhm_window(power, idx_peak, idx_min=0, pad_bins=0)`
The inclusive half-power band `(lo, hi)` of bin indices around a peak.

## Peaks, spectrograms, cross-spectra

### `spectral_peaks(data, *, n_peaks=3, dims=None, omega_min=1e-8, rel_height=1e-3, detrend=True, window=None)`
The strongest local maxima of a power spectrum, from a signal, a `time_fft`
Dataset or a power array (summed over `dims`). Returns a Dataset along `peak`
with `omega`, `omega_refined` (parabolic fit to the log power: sub-bin),
`power`, `omega_lo` and `omega_hi`. `detrend` may also be a polynomial degree
removed in `t` first, e.g. `2` for an energy, whose peaks sit at twice the
wave frequency.

### `spectrogram(data, *, length, step=None, detrend=True, window="hann")`
Power in sliding windows over `(t, omega, ...)`, with `t` at each window's
center. `length` and `step` are sample counts (int) or time spans (float);
`step` defaults to `length / 4`.

### `cross_spectrum(first, second, *, dims=None, detrend=True, window=None)`
Dataset of `cross` (`conj(F1) * F2`), `magnitude` and `phase` (radians, how
far `second` leads `first`). With `dims`, the values are summed over those
points and `coherence = |Σ cross| / Σ |cross|` (0 to 1) is added.

### `trace_branch(spectrum, theory, *, window=0.2, k_range=None, threshold=1e-3)`
The measured frequency of a dispersion branch near `theory(k)`, at every
`k ≥ 0` of an `(omega, k)` spectrum (from `array.struphy.analysis.dispersion()`).
The power at `+k` and `−k` is added, so waves moving either way count. The
branch may be curved. Returns a Dataset over `k` with `omega` (refined below
the bin spacing), `omega_theory` and `relative_error`. A `k` is NaN without a
local maximum in the window, or when that maximum is weaker than `threshold`
times the strongest one.

## Modes and eigenfunctions

### `mode_spectrum(data, *, dims=("eta2", "eta3"), names=("m", "n"), periods=1.0, scale=1)`
Complex Fourier amplitudes over integer mode numbers along periodic
directions, with a duplicate endpoint dropped first. Each direction must sample
a full period. `scale` multiplies the mode numbers, e.g. `(1, 6)` for
full-torus `n` of a sixth of a torus. Other dimensions are
kept, e.g. `(t, eta1, m, n)`.

### `mode_amplitudes(modes, *, top=None, real=True, relative=False)`
Stacks a mode spectrum along one labeled `mode` dimension, with coordinates
such as `"(10, -1)"` and `m`, `n`. For a real field, each conjugate pair is
merged and its amplitude doubled. `relative=True` divides by the mean (the
zero mode), which is then left out.

### `mode_structure(data, omega, *, window="hann", detrend=True)`
The complex amplitude at an exact frequency, at every point: `a·exp(iφ)` for a
field `a·cos(ωt + φ)`. For each harmonic's radial profile, use
`mode_spectrum(mode_structure(field, omega))`.

## Complex frequencies

### `matrix_pencil(data, *, n_modes=1, pencil=None, detrend=False)`
Fits a sum of growing or damped oscillations to a `(t,)` series (matrix-pencil
method), without the FFT's bin limit. Returns a Dataset along `mode`
(strongest first) with `omega`, `gamma`, `amplitude` and `phase`, and
`residual` in `attrs`. For a real signal, `n_modes` counts oscillations.

### `pencil_reconstruction(fit, t)`
The real signal described by a fit, at times `t`.

## Plots (`struphy_plots.spectral_plots`)

Each returns a `PlotResult`. The accessor forms take keyword selections of
every dimension except `t` (and those a plot keeps), e.g. `eta1=0.5`.

### `plot_power_spectrum(data, *, dims=None, detrend=True, window=None, peaks=None, band=None, frequencies=None, logy=True, dynamic_range=8.0, omega_max=None)`
Power per bin, averaged over `dims` (default: all but `component`), with one
line per remaining coordinate, labeled peaks, a shaded filter `band` and named
reference `frequencies`. Accessor: `plot.power_spectrum(...)`, and
`dataset.struphy.plot.power_spectrum()` for a `time_fft` Dataset.

### `plot_filtered(data, result, **selection)`
A probe of the signal, minus its mean, against its filtered reconstruction.
Accessor: `plot.filtered(result, eta1=..., ...)`.

### `plot_spectrogram(power, *, log=True, dynamic_range=4.0, omega_max=None, frequencies=None)`
Power over `(t, omega)`. Accessor: `plot.spectrogram(length=..., step=...)`,
which averages any other dimensions.

### `plot_mode_amplitudes(modes, *, top=6, fit=None, logy=True)`
The strongest mode amplitudes over time, with growth fits for `fit=(t0, t1)`
or `True`. Accessor: `plot.mode_amplitudes(top=..., fit=..., reduce="max")`.

### `plot_mode_map(modes, *, m_range=None, n_range=None, log=True)`
The absolute amplitude over the `(m, n)` plane. Accessor:
`plot.mode_map(t=-1, ...)`.

### `plot_radial_power(power, *, x="eta1", x_of=None, continuum=None, log=True, dynamic_range=3.0, omega_max=None)`
Power over `(omega, radius)`. `x_of` maps the coordinate to the plotted axis,
and `continuum` is a `(spectrum, modes)` pair or a prepared continuous
spectrum. Accessor: `plot.radial_power(...)`, which averages the angles.

### `plot_mode_profiles(structure, *, x="eta1", x_of=None, top=4, phase=True)`
Radial amplitude and phase of each harmonic; `(m, n)` and `(-m, -n)` count as
one. Accessor: `plot.mode_profiles(omega, ...)` for the eigenfunction at a
frequency, or `plot.mode_profiles(t=...)` for the amplitudes at one time; both
take `scale`.

### `plot_cross_spectrum(cross, *, omega_max=None)`
Magnitude, coherence (if present) and phase of a cross-spectrum. Accessor:
`plot.cross_spectrum(other, dims=...)`, and
`dataset.struphy.plot.cross_spectrum()`.

### `plot_pencil_fit(data, fit)`
The samples against a matrix-pencil fit, and the fitted modes in the
complex-frequency plane. Accessor: `plot.pencil_fit(n_modes=...)`.

## `out.analysis`

`struphy_plots.output_accessors.OutputAnalysis`, accessed as `out.analysis` on
a Struphy `Output`. `fft`, `time_fft`, `filter_time` and `mode_spectrum` take a
product name (evaluated with `out.evaluate`) or an array, with the same
options as above.
