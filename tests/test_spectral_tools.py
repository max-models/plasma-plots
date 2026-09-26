"""Tests for the spectral tools beyond the ported branch, their plots and entry points."""

import os

import h5py
import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

import struphy_plots  # noqa: E402, F401
from struphy_plots import spectral as sp  # noqa: E402
from struphy_plots import spectral_plots as spp  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def series(values, t, name="u"):
    return xr.DataArray(values, dims="t", coords={"t": t}, name=name, attrs={"label": name})


def torus_field(n_t=24, omega=0.4, growth=0.05):
    """(t, e1, e2, e3): an m=10 and an m=11 harmonic (n=-1) with radial envelopes, growing."""
    t = np.arange(n_t) * 0.5
    e1, e2, e3 = np.linspace(0, 1, 11), np.linspace(0, 1, 49), np.linspace(0, 1, 9)
    R, E2, E3 = np.meshgrid(e1, e2, e3, indexing="ij")
    values = np.stack(
        [
            np.exp(growth * ti)
            * (
                np.exp(-(((R - 0.4) / 0.15) ** 2)) * np.cos(2 * np.pi * (10 * E2 - E3) - omega * ti)
                + 0.5 * np.exp(-(((R - 0.6) / 0.15) ** 2)) * np.cos(2 * np.pi * (11 * E2 - E3) - omega * ti)
            )
            for ti in t
        ]
    )
    return xr.DataArray(
        values, dims=("t", "e1", "e2", "e3"), coords={"t": t, "e1": e1, "e2": e2, "e3": e3}, name="u",
        attrs={"label": "u", "run": "synthetic"},
    )


def test_drop_periodic_endpoint_only_drops_a_true_duplicate():
    closed = xr.DataArray(np.arange(5.0), dims="e2", coords={"e2": np.linspace(0, 1, 5)})
    assert sp.drop_periodic_endpoint(closed, "e2").sizes["e2"] == 4
    open_grid = closed.assign_coords(e2=np.arange(5) / 5)
    assert sp.drop_periodic_endpoint(open_grid, "e2").sizes["e2"] == 5
    with pytest.raises(ValueError, match="not a dimension"):
        sp.drop_periodic_endpoint(closed, "e3")


def test_band_filter_separates_two_on_bin_modes():
    t = np.arange(400) * 0.5
    slow, fast = np.sin(2 * np.pi * 5 * np.arange(400) / 400), 0.4 * np.cos(2 * np.pi * 12 * np.arange(400) / 400)
    data = series(slow + fast, t)
    omega_fast = 2 * np.pi * 12 / 200
    np.testing.assert_allclose(sp.band_filter(data, omega_fast * 0.9, omega_fast * 1.1), fast, atol=1e-12)
    np.testing.assert_allclose(data.struphy.analysis.band_filter(0.1, 0.2), slow, atol=1e-12)
    with pytest.raises(ValueError, match="exceed"):
        sp.band_filter(data, 1.0, 0.5)


def test_spectral_peaks_are_sorted_and_refined_below_the_bin_spacing():
    t = np.arange(400) * 0.5
    data = series(np.sin(0.7 * t) + 0.3 * np.sin(1.9 * t), t)
    peaks = sp.spectral_peaks(data, n_peaks=2, window="hann")
    resolution = peaks.attrs["frequency_resolution"]
    assert peaks.power.values[0] > peaks.power.values[1]
    np.testing.assert_allclose(peaks.omega_refined, [0.7, 1.9], atol=0.1 * resolution)
    assert (peaks.omega_lo <= peaks.omega).all() and (peaks.omega <= peaks.omega_hi).all()
    assert data.struphy.analysis.spectral_peaks(n_peaks=1).sizes["peak"] == 1


def test_spectrogram_follows_a_chirp_on_one_frequency_grid():
    t = np.arange(2000) * 0.1
    chirp = series(np.sin((0.5 + 0.005 * t) * t), t)
    power = sp.spectrogram(chirp, length=40.0)
    assert power.dims == ("t", "omega") and power.sizes["omega"] == 201
    first, last = (float(power.omega[int(power.isel(t=i).argmax("omega"))]) for i in (0, -1))
    instantaneous = lambda time: 0.5 + 0.01 * time  # noqa: E731
    assert first == pytest.approx(instantaneous(float(power.t[0])), abs=2 * power.attrs["frequency_resolution"])
    assert last == pytest.approx(instantaneous(float(power.t[-1])), abs=2 * power.attrs["frequency_resolution"])
    with pytest.raises(ValueError, match="length"):
        sp.spectrogram(chirp, length=3)


def test_mode_spectrum_and_amplitudes_find_the_seeded_harmonics():
    field = torus_field()
    modes = sp.mode_spectrum(field)
    assert modes.dims == ("t", "e1", "m", "n")
    assert modes.attrs["run"] == "synthetic"
    amplitudes = sp.mode_amplitudes(modes, top=2)
    assert list(amplitudes.mode.values) == ["(10, -1)", "(11, -1)"]
    assert list(amplitudes.m.values) == [10, 11]
    peak = amplitudes.isel(t=0).max("e1")
    np.testing.assert_allclose(peak, [1.0, 0.5], rtol=1e-6)
    assert field.struphy.analysis.mode_spectrum().equals(modes)


def test_mode_structure_recovers_amplitude_and_phase_at_an_off_bin_frequency():
    x = np.linspace(0, 1, 7)
    t = np.arange(300) * 0.2
    field = xr.DataArray(
        np.sin(np.pi * x)[None] * np.cos(0.83 * t[:, None] + 0.5 + x[None]),
        dims=("t", "e1"), coords={"t": t, "e1": x},
    )
    structure = sp.mode_structure(field, 0.83)
    assert structure.dims == ("e1",)
    np.testing.assert_allclose(abs(structure), np.sin(np.pi * x), atol=1e-3)
    inner = slice(1, -1)
    np.testing.assert_allclose(np.angle(structure.values[inner]), 0.5 + x[inner], atol=1e-3)


def test_cross_spectrum_phase_and_coherence():
    t = np.arange(256) * 0.25
    omega = 2 * np.pi * 8 / 64  # on a bin
    first = series(np.cos(omega * t), t, "u")
    second = series(-np.sin(omega * t), t, "b")
    cross = sp.cross_spectrum(first, second)
    peak = int(cross.magnitude.argmax("omega"))
    assert float(cross.phase[peak]) == pytest.approx(np.pi / 2)
    assert "coherence" not in cross

    rng = np.random.default_rng(1)
    x = np.arange(20)
    noisy = xr.DataArray(
        np.cos(omega * t)[:, None] + rng.normal(0, 0.1, (256, 20)), dims=("t", "e1"), coords={"t": t, "e1": x}
    )
    lagged = xr.DataArray(
        np.cos(omega * t - 1.0)[:, None] + rng.normal(0, 0.1, (256, 20)), dims=("t", "e1"), coords={"t": t, "e1": x}
    )
    averaged = noisy.struphy.analysis.cross_spectrum(lagged, dims="e1")
    peak = int(averaged.magnitude.argmax("omega"))
    assert float(averaged.coherence[peak]) > 0.95
    assert float(averaged.phase[peak]) == pytest.approx(-1.0, abs=0.05)


def test_matrix_pencil_resolves_frequencies_and_growth_from_a_short_record():
    t = np.linspace(0, 20, 41)  # a third of the slow mode's period
    signal = series(np.exp(0.02 * t) * np.cos(0.096 * t + 0.3) + 0.4 * np.exp(-0.1 * t) * np.cos(1.3 * t), t)
    fit = sp.matrix_pencil(signal, n_modes=2)
    np.testing.assert_allclose(fit.omega, [0.096, 1.3], rtol=1e-6)
    np.testing.assert_allclose(fit.gamma, [0.02, -0.1], atol=1e-8)
    np.testing.assert_allclose(fit.amplitude, [1.0, 0.4], rtol=1e-6)
    assert float(fit.phase[0]) == pytest.approx(0.3)
    assert fit.attrs["residual"] < 1e-10
    np.testing.assert_allclose(sp.pencil_reconstruction(fit, t), signal, atol=1e-10)
    assert signal.struphy.analysis.matrix_pencil(n_modes=2).omega.values == pytest.approx(fit.omega.values)

    complex_signal = series(np.exp((0.1 + 1j) * t), t)
    one = sp.matrix_pencil(complex_signal, n_modes=1)
    assert one.omega.item() == pytest.approx(1.0) and one.gamma.item() == pytest.approx(0.1)
    with pytest.raises(ValueError, match="cannot fit"):
        sp.matrix_pencil(signal.isel(t=slice(0, 6)), n_modes=2)
    with pytest.raises(ValueError, match="series"):
        sp.matrix_pencil(torus_field())


def test_power_spectrum_plot_with_peaks_band_and_reference_lines():
    t = np.arange(400) * 0.5
    data = series(np.sin(0.7 * t) + 0.3 * np.sin(1.9 * t) + 0.2, t)
    band = sp.filter_time(data)
    result = data.struphy.plot.power_spectrum(peaks=2, band=band, frequencies={"theory": 0.7})
    assert result.data["peaks"].sizes["peak"] == 2
    labels = [text.get_text() for text in result.ax.get_legend().get_texts()]
    assert {"peaks", "filter band", "theory"} <= set(labels)
    assert data.struphy.analysis.time_fft().struphy.plot.power_spectrum().ax is not None

    field = torus_field()
    per_radius = spp.plot_power_spectrum(field, dims=("e2", "e3"), omega_max=2.0)
    assert len(per_radius.artists) == field.sizes["e1"]
    with pytest.raises(ValueError, match="single line"):
        spp.plot_power_spectrum(field, dims=("e2", "e3"), peaks=1)


def test_mode_plots_on_a_torus_field():
    field = torus_field()
    amplitudes = field.struphy.plot.mode_amplitudes(top=2, fit=True)
    assert [fit.rate for fit in amplitudes.fit_results] == pytest.approx([0.05, 0.05], rel=1e-6)
    assert field.struphy.plot.mode_map(t="last", m_range=(0, 15), n_range=(-4, 4)).data["amplitude"].dims == ("n", "m")
    profiles = field.struphy.plot.mode_profiles(0.4, top=2)
    peaks = profiles.data["profiles"]
    assert abs(peaks).idxmax("e1").values.tolist() == pytest.approx([0.4, 0.6])
    radial = field.struphy.plot.radial_power(
        x_of=lambda e1: 0.1 + 0.9 * e1, continuum=(lambda r, m, n: {"alfven": np.abs(n + m / (1 + r))}, [(1, 0)]),
        omega_max=2.0,
    )
    assert radial.data["power"].dims == ("omega", "e1")
    assert len(radial.artists) == 2


def test_spectrogram_filtered_cross_and_pencil_plots():
    t = np.arange(800) * 0.1
    data = series(np.sin((0.5 + 0.01 * t) * t), t)
    assert data.struphy.plot.spectrogram(length=20.0, frequencies={"start": 0.5}).data["spectrogram"].dims == ("omega", "t")
    field = torus_field(n_t=64)
    result = field.struphy.analysis.filter_time(dims=("e1", "e2", "e3"))
    probe = field.struphy.plot.filtered(result, e1=0.4, e2=0.0, e3=0.0)
    assert len(probe.artists) == 2
    u, b = series(np.cos(0.5 * t), t, "u"), series(-np.sin(0.5 * t), t, "b")
    assert u.struphy.plot.cross_spectrum(b).data["peak_phase_deg"] == pytest.approx(90, abs=5)
    short = series(np.exp(0.05 * t[:60]) * np.cos(0.3 * t[:60]), t[:60])
    fit = short.struphy.plot.pencil_fit(n_modes=1)
    assert fit.data["fit"].gamma.item() == pytest.approx(0.05, rel=1e-6)


def test_output_analysis_matches_the_array_accessor(tmp_path):
    from struphy.post_processing.output import Output
    from struphy.post_processing.tests.test_output import write_manifest, write_tree

    path = os.path.join(tmp_path, "sim_1")
    os.makedirs(path)
    write_tree(path)
    with h5py.File(os.path.join(path, "data", "data_proc0.hdf5"), "a") as file:
        time = np.asarray(file["time/value"])
        file.create_dataset("scalar/en_phi", data=np.cos(time))
    write_manifest(path)
    out = Output(path)
    field = out.fields.em_fields.E.isel(component=0, e2=0, e3=0)
    xr.testing.assert_identical(out.analysis.time_fft(field), field.struphy.analysis.time_fft())
    xr.testing.assert_identical(out.analysis.fft(field, dim="e1"), field.struphy.analysis.fft(dim="e1"))
    xr.testing.assert_identical(out.analysis.filter_time(field).filtered, field.struphy.analysis.filter_time().filtered)
    by_name = out.analysis.time_fft("em_fields/E")
    assert "omega" in by_name.dims and "t" not in by_name.dims
