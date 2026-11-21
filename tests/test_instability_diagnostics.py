"""Growth detection, the dominant mode, the Penrose criterion and the electrostatic growth of drifting Maxwellians."""

import numpy as np
import pytest
import xarray as xr

import plasma_plots  # noqa: F401 -- registers the accessors
from plasma_plots.analysis import (
    GrowthFit,
    detect_growth,
    dominant_mode,
    growth_rate,
    penrose_criterion,
)
from plasma_plots.theory.kinetic import (
    Maxwellian,
    bump_on_tail,
    electrostatic_dielectric,
    electrostatic_growth,
    langmuir,
    maximum_growth,
    two_stream,
)

TIME = np.linspace(0.0, 120.0, 1201)


def _series(values, time=TIME):
    return xr.DataArray(values, coords={"t": time}, dims="t")


def _saturating(rate=0.4, noise=1e-5, rng=None):
    """Logistic growth to 1, half-way at t = 80, over a noise floor of random phase (adding in power)."""
    grow = np.exp(rate * (TIME - 80.0))
    floor = noise if rng is None else noise * (1 + 0.1 * rng.standard_normal(TIME.size))
    return _series(np.sqrt((grow / (1 + grow)) ** 2 + floor**2))


# ---------------------------------------------------------------------------------------------
# detect_growth
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [0.2, 0.4, 1.0])
def test_growth_rate_from_noise_to_saturation(rate):
    rng = np.random.default_rng(0)
    phase = detect_growth(_saturating(rate, rng=rng))
    assert phase is not None
    assert phase.rate == pytest.approx(rate, rel=0.05)
    assert phase.efolds > 4
    assert phase.r2 > 0.99
    assert phase.noise == pytest.approx(1e-5, rel=0.3)
    assert phase.peak == pytest.approx(1.0, rel=0.01)
    assert phase.start < phase.stop
    assert phase.fitted.shape == phase.time.shape


def test_energy_gives_the_rate_of_the_amplitude():
    energy = _saturating(0.4) ** 2
    phase = detect_growth(energy, amplitude_from_quadratic=True)
    assert phase.rate == pytest.approx(0.4, rel=0.05)
    assert phase.noise == pytest.approx(1e-10, rel=0.3)
    assert phase.peak == pytest.approx(1.0, rel=0.01)
    given = detect_growth(energy, amplitude_from_quadratic=True, noise=1e-10)
    assert given.rate == pytest.approx(phase.rate, rel=0.01)


@pytest.mark.parametrize(
    "values",
    [
        np.full(TIME.size, 1e-3),  # constant
        1e-3
        * (1 + 0.3 * np.random.default_rng(1).standard_normal(TIME.size)) ** 2,  # noise
        np.exp(-0.1 * TIME),  # damped
        1e-3 * (1 + 0.01 * TIME),  # slow linear drift: not 2 e-foldings
    ],
    ids=["constant", "noise", "damped", "drift"],
)
def test_nothing_grows(values):
    assert detect_growth(_series(values)) is None


def test_decay_before_growth_starts_from_the_minimum():
    # a seeded wave that damps, then an instability that grows from the noise
    values = np.exp(-0.5 * TIME) + 1e-8 * np.exp(0.5 * TIME)
    phase = detect_growth(_series(values), min_r2=0.9)
    assert phase is not None
    assert phase.rate == pytest.approx(0.5, rel=0.05)
    assert phase.start >= TIME[np.argmin(values)]


def test_oscillating_amplitude_with_peaks_only():
    values = np.abs(np.cos(1.3 * TIME)) * 1e-6 * np.exp(0.3 * TIME)
    values = values / (1 + values) + 1e-6
    assert detect_growth(_series(values)) is None
    phase = detect_growth(_series(values), peaks_only=True)
    assert phase.rate == pytest.approx(0.3, rel=0.05)


def test_short_phases_and_too_few_samples_are_rejected():
    assert detect_growth(_series([1.0, 2.0, 4.0], TIME[:3])) is None
    assert detect_growth(_saturating(0.4), min_efolds=50) is None
    assert detect_growth(_saturating(0.4), stop=1e-6) is None


def test_agrees_with_growth_rate_on_its_window():
    series = _saturating(0.4)
    phase = detect_growth(series)
    fit = growth_rate(series, GrowthFit(window=(phase.start, phase.stop)))
    # the same window; growth_rate keeps the noise floor in, detect_growth takes it out
    assert fit.rate == pytest.approx(phase.rate, rel=0.02)


def test_detect_growth_needs_t():
    with pytest.raises(ValueError, match="dims"):
        detect_growth(xr.DataArray(np.ones((3, 3)), dims=("t", "x")))


# ---------------------------------------------------------------------------------------------
# dominant_mode
# ---------------------------------------------------------------------------------------------

X = np.linspace(0.0, 4.0, 128, endpoint=False)
T = np.linspace(0.0, 30.0, 600)


def _wave(k, omega, amplitude=1.0):
    return amplitude * np.cos(k * X[None] - omega * T[:, None])


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_travelling_wave(sign):
    k = 2 * np.pi * 3 / 4.0
    data = xr.DataArray(_wave(sign * k, 2.0), coords={"t": T, "x": X})
    mode = dominant_mode(data)
    assert mode.k == pytest.approx(k)
    assert mode.omega == pytest.approx(2.0, rel=0.02)
    assert mode.phase_velocity == pytest.approx(sign * 2.0 / k, rel=0.02)
    assert mode.wavelength == pytest.approx(4.0 / 3)
    assert mode.power_fraction > 0.9


def test_strongest_of_two_waves_and_a_static_part():
    k1, k2 = 2 * np.pi / 4.0, 2 * np.pi * 5 / 4.0
    static = 3.0 * np.sin(k1 * X)[None] * np.ones_like(T)[:, None]
    data = xr.DataArray(
        _wave(k1, 0.7, 0.2) + _wave(-k2, 3.0) + static, coords={"t": T, "x": X}
    )
    mode = data.plasma.analysis.dominant_mode(dim="x")
    assert mode.k == pytest.approx(k2)
    assert mode.phase_velocity == pytest.approx(-3.0 / k2, rel=0.02)


def test_dominant_mode_without_a_usable_frequency():
    data = xr.DataArray(_wave(1.0, 2.0), coords={"t": T, "x": X})
    assert dominant_mode(data, min_periods=1e6) is None
    assert (
        dominant_mode(xr.DataArray(np.zeros((T.size, X.size)), coords={"t": T, "x": X}))
        is None
    )
    single = xr.DataArray(np.ones((1, X.size)), coords={"t": [0.0], "x": X})
    assert dominant_mode(single) is None


def test_dominant_mode_dims():
    data = xr.DataArray(np.zeros((3, 4, 5)), dims=("t", "x", "y"))
    with pytest.raises(ValueError, match="dim is required"):
        dominant_mode(data)
    with pytest.raises(ValueError, match="exactly"):
        dominant_mode(data, dim="x")


# ---------------------------------------------------------------------------------------------
# penrose_criterion
# ---------------------------------------------------------------------------------------------

V = np.linspace(-10.0, 10.0, 801)


def _maxwellian(n=1.0, u=0.0, vt=1.0):
    return n / (np.sqrt(2 * np.pi) * vt) * np.exp(-((V - u) ** 2) / (2 * vt**2))


def _f(values):
    return xr.DataArray(values, coords={"v": V}, dims="v")


def test_single_maxwellian_is_stable():
    result = penrose_criterion(_f(_maxwellian()))
    assert not result.unstable
    assert np.isnan(result.velocity)
    assert result.minima.size == 0


@pytest.mark.parametrize(
    ("separation", "unstable"),
    # two equal warm streams at ±u: unstable for u > 1.3 v_t (Penrose; Krall and Trivelpiece 9.7)
    [(1.2, False), (1.5, True), (3.0, True)],
)
def test_two_streams(separation, unstable):
    f = _maxwellian(0.5, separation) + _maxwellian(0.5, -separation)
    result = _f(f).plasma.analysis.penrose()
    assert result.unstable is unstable


def test_bump_on_tail_agrees_with_the_dispersion_relation():
    # a cold beam makes a bump; a warm, slow one only a shoulder
    for density, drift, spread, unstable in (
        (0.1, 4.5, 0.5, True),
        (0.1, 3.0, 1.5, False),
    ):
        f = _maxwellian(1 - density) + _maxwellian(density, drift, spread)
        result = penrose_criterion(_f(f), prominence=1e-4)
        species = (
            Maxwellian(1 - density),
            Maxwellian(density, thermal_speed=spread, drift=drift),
        )
        growth = maximum_growth(
            lambda k, s=species: electrostatic_growth(k, s), (0.05, 0.8)
        )[1].imag
        assert result.unstable is unstable
        assert (growth > 1e-10) is unstable
        if unstable:
            assert 3.0 < result.velocity < 4.5


def test_noise_dips_are_not_minima():
    rng = np.random.default_rng(2)
    samples = rng.standard_normal(200_000)
    counts, edges = np.histogram(samples, bins=120, range=(-6, 6))
    centres = 0.5 * (edges[1:] + edges[:-1])
    f = xr.DataArray(counts.astype(float), coords={"v": centres}, dims="v")
    assert not penrose_criterion(f).unstable


def test_penrose_dims():
    with pytest.raises(ValueError, match="one-dimensional"):
        penrose_criterion(xr.DataArray(np.ones((2, 2)), dims=("v", "w")))
    with pytest.raises(ValueError, match="not the dimension"):
        penrose_criterion(_f(_maxwellian()), dim="w")


# ---------------------------------------------------------------------------------------------
# electrostatic_growth
# ---------------------------------------------------------------------------------------------


def test_two_stream_root():
    beams = (
        Maxwellian(0.5, thermal_speed=0.3, drift=3.0),
        Maxwellian(0.5, thermal_speed=0.3, drift=-3.0),
    )
    k = np.linspace(0.05, 0.3, 6)
    np.testing.assert_allclose(
        np.asarray(electrostatic_growth(k, beams)).imag,
        np.asarray(two_stream(k, 3.0, 0.3)).imag,
        atol=1e-8,
    )


def test_bump_on_tail_root():
    species = (Maxwellian(0.9), Maxwellian(0.1, thermal_speed=0.5, drift=4.5))
    k = np.linspace(0.1, 0.4, 5)
    np.testing.assert_allclose(
        electrostatic_growth(k, species), bump_on_tail(k), rtol=1e-8
    )


def test_stable_maxwellian_gives_the_landau_root():
    assert electrostatic_growth(0.3, Maxwellian()) == pytest.approx(
        langmuir(0.3), rel=1e-8
    )


def test_buneman_growth_near_the_cold_maximum():
    mass_ratio = 1836.15
    species = (Maxwellian(drift=20.0), Maxwellian.ions(mass_ratio=mass_ratio))
    k, omega = maximum_growth(lambda k: electrostatic_growth(k, species), (0.02, 0.06))
    cold = np.sqrt(3) / 2 * (1 / (2 * mass_ratio)) ** (1 / 3)
    assert k * 20.0 == pytest.approx(1.0, abs=0.1)
    assert omega.imag == pytest.approx(cold, rel=0.1)
    assert abs(electrostatic_dielectric(omega, k, species)) < 1e-8


def test_ion_acoustic_instability_with_a_drift():
    ions = Maxwellian.ions(mass_ratio=100.0, temperature_ratio=20.0)
    assert electrostatic_growth(0.3, (Maxwellian(drift=1.5), ions)).imag > 0
    assert electrostatic_growth(0.3, (Maxwellian(drift=0.0), ions)).imag < 0


def test_extra_guesses():
    species = (Maxwellian(0.9), Maxwellian(0.1, thermal_speed=0.5, drift=4.5))
    omega = electrostatic_growth([0.3], species, guesses=[1.0 + 0.2j])
    assert omega[0] == pytest.approx(bump_on_tail(0.3), rel=1e-8)
