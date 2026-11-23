"""plasma_plots.theory.negative_ions against independent limits, identities and numerical integrals."""

import numpy as np
import pytest

from plasma_plots.theory import beams, parameters, sheaths
from plasma_plots.theory import negative_ions as ni
from plasma_plots.theory.parameters import electron_mass, elementary_charge, proton_mass


def test_cross_section_fits_against_values_quoted_in_the_source():
    # Janev et al. section 3.1.2: sigma_det ~ 2.3e-15 cm^2 at 65 eV (the fit, eq. 58, gives 2.2)
    assert ni.electron_detachment_cross_section(65.0) * 1e4 == pytest.approx(
        2.3e-15, rel=0.06
    )
    # zero at and below the 0.754 eV binding energy, positive above
    assert ni.electron_detachment_cross_section([0.1, 0.754]).tolist() == [0.0, 0.0]
    assert ni.electron_detachment_cross_section(1.0) > 0
    # H- + H2 (eq. 159): maximum of about 1e-15 cm^2 near 2 keV, threshold 2.18 eV
    energy = np.geomspace(2.0, 1e5, 400)
    sigma = ni.h2_detachment_cross_section(energy) * 1e4
    assert 1.0e-15 < sigma.max() < 1.4e-15
    assert 1e3 < energy[sigma.argmax()] < 4e3
    assert ni.h2_detachment_cross_section(2.0) < 1e-40
    # the detachment channel of H- + H (eq. 71) dominates the associative one above 1 eV
    assert ni.hydrogen_detachment_cross_section(
        10.0
    ) > ni.hydrogen_detachment_cross_section(10.0, "associative")


def test_mutual_neutralization_levels_and_range():
    total = ni.mutual_neutralization_cross_section(np.array([0.1, 1.0, 100.0, 2e4]))
    parts = sum(
        ni.mutual_neutralization_cross_section(np.array([0.1, 1.0, 100.0, 2e4]), n)
        for n in (2, 3)
    )
    assert np.allclose(total, parts)
    # about 1e-13 cm^2 scale at thermal energies, decreasing steeply at high energy
    assert 1e-15 < ni.mutual_neutralization_cross_section(1.0) * 1e4 < 1e-12
    assert total[-1] < 0.1 * total[1]
    with pytest.raises(ValueError):
        ni.mutual_neutralization_cross_section(1.0, 4)
    with pytest.raises(ValueError):
        ni.hydrogen_detachment_cross_section(1.0, "x")


def test_maxwellian_average_of_power_laws():
    # constant sigma: <sigma v> = sigma sqrt(8 T / (pi mu))
    mu = electron_mass * proton_mass / (electron_mass + proton_mass)
    for t in (0.5, 3.0):
        exact = 1e-19 * np.sqrt(8 * t * elementary_charge / (np.pi * mu))
        assert ni.maxwellian_rate_coefficient(
            lambda e: 1e-19 + 0 * e, t
        ) == pytest.approx(exact, rel=2e-3)
    # sigma = c / v (c / sqrt(E) up to constants): <sigma v> is independent of temperature
    c = 1e-20
    rates = ni.maxwellian_rate_coefficient(
        lambda e: c / np.sqrt(e),
        np.array([0.5, 2.0, 8.0]),
        mass_a=proton_mass,
        mass_b=proton_mass,
    )
    assert rates == pytest.approx(
        c * np.sqrt(2 * elementary_charge / (proton_mass / 2)), rel=2e-3
    )
    # the relative temperature of two Maxwellians of equal masses is the mean
    f = lambda e: 1e-19 + 0 * e  # noqa: E731
    a = ni.maxwellian_rate_coefficient(
        f, 1.0, proton_mass, proton_mass, temperature_b=3.0
    )
    b = ni.maxwellian_rate_coefficient(f, 2.0, proton_mass, proton_mass)
    assert a == pytest.approx(b, rel=1e-6)
    # arrays keep their shape
    assert ni.electron_detachment_rate_coefficient(np.ones((2, 3))).shape == (2, 3)


def test_thermal_rates_have_the_expected_behaviour():
    # threshold 0.754 eV: the detachment rate is exponentially small for T << 0.754 eV and
    # approaches a few 1e-13 m^3/s (1e-7 cm^3/s) at 5 eV
    t = np.array([0.2, 0.5, 1.0, 2.0, 5.0])
    rate = ni.electron_detachment_rate_coefficient(t)
    assert np.all(np.diff(rate) > 0)
    assert rate[0] < 1e-3 * rate[-1]
    assert 1e-14 * 1e-6 < rate[-1] < 1e-12
    # mutual neutralization: ~1e-7 cm^3/s = 1e-13 m^3/s, weakly decreasing with temperature
    mn = ni.mutual_neutralization_rate_coefficient(np.array([0.5, 1.0, 2.0]))
    assert np.all(np.diff(mn) < 0)
    assert np.all((2e-14 < mn) & (mn < 2e-13))


def test_destruction_frequencies_are_densities_times_rates():
    nu = ni.destruction_frequencies(
        1e18, 2.0, positive_ion_density=5e17, ion_temperature=1.0, atom_density=1e19
    )
    assert nu["electron detachment"] == pytest.approx(
        1e18 * ni.electron_detachment_rate_coefficient(2.0, 1.0)
    )
    assert nu["mutual neutralization"] == pytest.approx(
        5e17 * ni.mutual_neutralization_rate_coefficient(1.0)
    )
    assert nu["atom detachment"] > 0
    assert ni.destruction_frequencies(1e18, 2.0)["atom detachment"] == 0.0
    # scales linearly with the density
    nu2 = ni.destruction_frequencies(
        2e18, 2.0, positive_ion_density=1e18, ion_temperature=1.0, atom_density=2e19
    )
    assert all(nu2[k] == pytest.approx(2 * nu[k]) for k in nu)


def test_debye_length_limits():
    n, te = 1e18, 2.0
    # alpha -> 0: the electron Debye length
    assert ni.debye_length_electronegative(n, te, 0.0, 0.5) == pytest.approx(
        parameters.debye_length(n, te)
    )
    # H- only adds shielding: shorter than the electron length, and shorter for colder H-
    cold = ni.debye_length_electronegative(n, te, 5.0, 0.2)
    warm = ni.debye_length_electronegative(n, te, 5.0, 2.0)
    assert cold < warm < parameters.debye_length(n, te)
    # all temperatures equal with positive ions: lambda_e / sqrt(2 (1 + alpha))
    for alpha in (0.0, 3.0, 100.0):
        got = ni.debye_length_electronegative(n, te, alpha, te, te)
        assert got == pytest.approx(
            parameters.debye_length(n, te) / np.sqrt(2 * (1 + alpha))
        )
    # an ion-ion plasma (n_e << n-): lambda = sqrt(eps0 T / (2 n- e^2)) for equal temperatures
    n_minus = 1e18
    lam = ni.debye_length_electronegative(1e14, 0.5, n_minus / 1e14, 0.5, 0.5)
    assert lam == pytest.approx(
        np.sqrt(
            parameters.vacuum_permittivity
            * 0.5
            * elementary_charge
            / (2 * n_minus * elementary_charge**2)
        ),
        rel=1e-3,
    )


def test_bohm_speed_limits():
    te = np.array([1.0, 5.0])
    # alpha -> 0 recovers the electropositive Bohm speed
    assert ni.bohm_speed_electronegative(te, 0.0, 0.5) == pytest.approx(
        sheaths.bohm_speed(te)
    )
    # equal temperatures (gamma = 1): independent of alpha
    assert ni.bohm_speed_electronegative(5.0, 7.0, 5.0) == pytest.approx(
        sheaths.bohm_speed(5.0)
    )
    # gamma > 1: slower, decreasing with alpha, and u_B^2 -> T_e / (gamma M) = T- / M as alpha -> inf
    speeds = ni.bohm_speed_electronegative(5.0, np.array([0.0, 1.0, 10.0, 1e6]), 1.0)
    assert np.all(np.diff(speeds) < 0)
    assert speeds[-1] == pytest.approx(sheaths.bohm_speed(1.0), rel=1e-3)
    # the closed form and the equivalent (n- + ne) T- / (n- Te + ne T-) of Paul et al.
    alpha, tm, t = 2.0, 0.5, 3.0
    ratio = (alpha + 1) * tm / (alpha * t + tm)
    assert ni.bohm_speed_electronegative(t, alpha, tm) ** 2 == pytest.approx(
        ratio * sheaths.bohm_speed(t) ** 2
    )
    # heavier ions: 1 / sqrt(M)
    assert ni.bohm_speed_electronegative(5.0, 1.0, 1.0, mass_number=4) == pytest.approx(
        0.5 * ni.bohm_speed_electronegative(5.0, 1.0, 1.0)
    )


def test_sheath_edge_electronegativity_and_flux():
    assert ni.sheath_edge_electronegativity(5.0, 3.0, 2.0, 2.0) == pytest.approx(5.0)
    assert ni.sheath_edge_electronegativity(5.0, 0.0, 2.0, 0.5) == pytest.approx(5.0)
    # Boltzmann relation: n- ~ n_e^gamma
    te, tm, alpha0, v = 3.0, 0.5, 4.0, 1.5
    ne_ratio = np.exp(-v / te)
    expected = alpha0 * ne_ratio ** (te / tm - 1)
    assert ni.sheath_edge_electronegativity(alpha0, v, te, tm) == pytest.approx(
        expected
    )
    # flux: (1 + alpha) n_e u_B, and the electropositive n u_B at alpha = 0
    assert ni.ion_flux_electronegative(1e17, 0.0, 5.0, 1.0) == pytest.approx(
        1e17 * sheaths.bohm_speed(5.0)
    )
    flux = ni.ion_flux_electronegative(1e17, 2.0, 5.0, 5.0)
    assert flux == pytest.approx(3 * 1e17 * sheaths.bohm_speed(5.0))


def test_beam_survival_and_surface_current():
    sigma = ni.h2_detachment_cross_section(30e3)
    n, length = 1e18, 0.3
    assert ni.beam_survival_fraction(
        30e3, h2_density=n, length=length
    ) == pytest.approx(beams.neutral_beam_attenuation(n, sigma, length=length))
    assert ni.beam_survival_fraction(30e3, length=length) == 1.0
    # two gases multiply; a sampled path agrees with a constant one
    x = np.linspace(0.0, length, 11)
    both = ni.beam_survival_fraction(
        30e3, 1e18, atom_density=5e18, positions=x, axis=-1
    )
    atoms = ni.beam_survival_fraction(30e3, 0.0, length=length, atom_density=5e18)
    mols = ni.beam_survival_fraction(30e3, 1e18, length=length)
    assert both == pytest.approx(atoms * mols)
    # more gas, less beam; stripping is the complement
    assert ni.beam_survival_fraction(30e3, 2e18, length=length) < mols
    assert ni.surface_production_current(1e22, 0.05, 0.7, 1e-4) == pytest.approx(
        1e22 * 0.05 * 0.7 * 1e-4 * elementary_charge
    )
    assert ni.surface_production_current(np.array([1e21, 2e21]), 0.1).shape == (2,)
