"""plasma_plots.theory.beams against independent identities, limits and numerical integrals."""

import numpy as np
import pytest

from plasma_plots.theory import beams
from plasma_plots.theory.parameters import (
    elementary_charge,
    electron_mass,
    plasma_frequency,
    proton_mass,
    speed_of_light,
    vacuum_permittivity,
)


def test_characteristic_current_and_perveance():
    # I0 = 4 pi eps0 m c^3 / q: 17.045 kA for electrons, 31.3 MA for protons (Reiser)
    for mass, i0 in [(electron_mass, 17.045e3), (proton_mass, 31.30e6)]:
        v, _ = beams.beam_speed(1e3, mass=mass)
        k = beams.perveance(1.0, 1e3, mass=mass, kind="generalized")
        assert k == pytest.approx(2 * 1.0 / (i0 * (v / speed_of_light) ** 3), rel=2e-3)
    # the relativistic K carries 1/(beta gamma)^3
    k_rel = beams.perveance(1.0, 1e6, kind="generalized", relativistic=True)
    v, gamma = beams.beam_speed(1e6, relativistic=True)
    assert k_rel == pytest.approx(2 / (17.045e3 * (v / speed_of_light) ** 3 * gamma**3), rel=2e-3)
    assert gamma == pytest.approx(1 + 1e6 * elementary_charge / (electron_mass * speed_of_light**2))
    assert beams.perveance([1.0, 2.0], 100.0).shape == (2,)
    with pytest.raises(ValueError):
        beams.perveance(1.0, 1.0, kind="x")


def test_k_from_density_and_plasma_frequency():
    # uniform beam of radius a: K = omega_p^2 a^2 / (2 v^2), from I = n q v pi a^2
    a, voltage, n = 3e-3, 20e3, 1e15
    v, _ = beams.beam_speed(voltage, mass=proton_mass)
    current = n * elementary_charge * v * np.pi * a**2
    k = beams.perveance(current, voltage, mass=proton_mass, kind="generalized")
    wp = beams.beam_plasma_frequency(n, mass=proton_mass)
    assert wp == pytest.approx(plasma_frequency(n, mass=proton_mass))
    assert k == pytest.approx(wp**2 * a**2 / (2 * v**2), rel=1e-12)
    assert beams.beam_plasma_frequency(n, gamma=2.0) == pytest.approx(plasma_frequency(n) / 2**1.5)
    assert beams.beam_plasma_frequency(n, gamma=2.0, direction="transverse") == pytest.approx(
        plasma_frequency(n) / 2**0.5
    )


def test_child_langmuir():
    # textbook: J = 2.334e-6 V^1.5/d^2 A/m^2 for electrons
    j = beams.child_langmuir_aperture(1.0, np.sqrt(1 / np.pi), 1.0)
    assert j == pytest.approx(2.334e-6, rel=1e-3)
    # ions scale as 1/sqrt(mass); current scales as V^1.5 r^2/d^2
    ratio = beams.child_langmuir_aperture(1.0, 1.0, 1.0, mass=proton_mass) / beams.child_langmuir_aperture(
        1.0, 1.0, 1.0
    )
    assert ratio == pytest.approx(np.sqrt(electron_mass / proton_mass))
    assert beams.child_langmuir_aperture(4.0, 1.0, 1.0) == pytest.approx(
        8 * beams.child_langmuir_aperture(1.0, 1.0, 1.0)
    )
    # the optimum perveance is the CL one
    voltage, r, d = 25e3, 2e-3, 3e-3
    assert beams.optimal_perveance(r / d, mass=proton_mass) == pytest.approx(
        beams.child_langmuir_aperture(voltage, r, d, mass=proton_mass) / voltage**1.5
    )
    # check against Child's law by integrating the 1-D Poisson equation: phi'' = J/(eps0 v)
    voltage, gap = 1e3, 1e-2
    j = 4 * vacuum_permittivity / 9 * np.sqrt(2 * elementary_charge / electron_mass) * voltage**1.5 / gap**2
    # phi = V (x/d)^(4/3) has phi'' = (4/9) V d^(-4/3) x^(-2/3) and v = sqrt(2 e phi/m)
    x = np.linspace(0.1, 1.0, 5) * gap
    phi = voltage * (x / gap) ** (4 / 3)
    phi_xx = 4 / 9 * voltage * gap ** (-4 / 3) * x ** (-2 / 3)
    speed = np.sqrt(2 * elementary_charge * phi / electron_mass)
    np.testing.assert_allclose(phi_xx, j / (vacuum_permittivity * speed), rtol=1e-12)


def test_envelope_drift_closed_form_and_hyperbola():
    # waist: x^2 = x0^2 + (eps z / x0)^2
    z = np.linspace(0, 5, 6)
    x = beams.kv_envelope(z, 2e-3, 0.0, 4e-6)
    np.testing.assert_allclose(x**2, (2e-3) ** 2 + (4e-6 * z / 2e-3) ** 2)
    # the closed form solves x'' = eps^2/x^3 (finite differences)
    h = 1e-3
    zz = 1.3
    xs = [beams.kv_envelope(zz + d, 1e-3, 2e-4, 1e-6) for d in (-h, 0.0, h)]
    xpp = (xs[0] - 2 * xs[1] + xs[2]) / h**2
    assert xpp == pytest.approx(1e-6**2 / xs[1] ** 3, rel=1e-5)
    # zero perveance, zero focusing: the RK4 integrator reproduces it, including a tilted start
    radius, slope = beams.kv_envelope_ode(z, 1e-3, 2e-4, 1e-6)
    np.testing.assert_allclose(radius, beams.kv_envelope(z, 1e-3, 2e-4, 1e-6), rtol=1e-7)


def test_envelope_zero_emittance_space_charge_expansion():
    # the first integral gives x'^2 = (K/2) ln(x/x0); the distance is
    # z = a0 sqrt(2/k) int_0^s exp(t^2) dt with s = sqrt(ln(a/a0)), k = K/4 for x''=k/x
    k_perv, x0 = 1e-3, 1e-3
    z = np.linspace(0, 3, 4)
    radius, slope = beams.kv_envelope_ode(z, x0, 0.0, 0.0, perveance=k_perv)
    s = np.sqrt(np.log(radius / x0))
    # independent quadrature of exp(t^2)
    t = np.linspace(0, 1, 20001)
    integral = np.array([np.trapezoid(np.exp((t * si) ** 2) * si, t) for si in s])
    np.testing.assert_allclose(x0 * np.sqrt(2 / (k_perv / 4)) * integral, z, rtol=1e-6, atol=1e-12)
    # the same through the first integral x'^2 = (K/2) ln(x/x0)
    np.testing.assert_allclose(slope**2, k_perv / 2 * np.log(radius / x0), rtol=1e-5, atol=1e-14)


def test_envelope_invariants_and_focusing():
    # drift with emittance and space charge: x'^2 = s0^2 + eps^2 (1/x0^2 - 1/x^2) + (K/2) ln(x/x0)
    k_perv, eps, x0, s0 = 2e-4, 2e-6, 1e-3, -1e-4
    z = np.linspace(0, 10, 11)
    radius, slope = beams.kv_envelope_ode(z, x0, s0, eps, perveance=k_perv)
    first_integral = s0**2 + eps**2 * (1 / x0**2 - 1 / radius**2) + k_perv / 2 * np.log(radius / x0)
    np.testing.assert_allclose(slope**2, first_integral, rtol=1e-6, atol=1e-16)
    # matched beam in a constant channel: x'' = 0 when kappa x = eps^2/x^3 + K/(4x)
    x_match = 1e-3
    kappa = eps**2 / x_match**4 + k_perv / (4 * x_match**2)
    radius, _ = beams.kv_envelope_ode(z, x_match, 0.0, eps, perveance=k_perv, focusing=lambda zz: kappa)
    np.testing.assert_allclose(radius, x_match, rtol=1e-9)
    # a weak focusing channel without space charge oscillates with the betatron wavenumber
    # sqrt(kappa) for small amplitude about the matched radius: period 2 pi / (2 sqrt(kappa)) for x
    kappa, eps = 25.0, 1e-6
    x_m = np.sqrt(eps / np.sqrt(kappa))
    zs = np.linspace(0, np.pi / np.sqrt(kappa), 3)
    radius, _ = beams.kv_envelope_ode(zs, 1.01 * x_m, 0.0, eps, focusing=lambda zz: kappa)
    assert radius[2] == pytest.approx(radius[0], rel=1e-5)  # envelope period is pi/sqrt(kappa)
    assert radius[1] < radius[0]
    with pytest.raises(ValueError):
        beams.kv_envelope_ode([1.0, 0.0], 1e-3, 0.0, 1e-6)


def test_divergence_and_pierce():
    assert beams.beam_divergence_gaussian(1.0, 100.0) == pytest.approx(0.1)
    assert beams.beam_divergence_gaussian(2.0, 100.0, "rms") == pytest.approx(0.1)
    # rms divergence equals the rms transverse over axial speed of a sampled Maxwellian
    rng = np.random.default_rng(0)
    m, t_ev, e_ev = proton_mass, 5.0, 2e3
    vperp = rng.normal(0, np.sqrt(t_ev * elementary_charge / m), 400_000)
    vz = np.sqrt(2 * e_ev * elementary_charge / m)
    assert np.std(vperp / vz) == pytest.approx(beams.beam_divergence_gaussian(t_ev, e_ev, "rms"), rel=5e-3)
    assert beams.pierce_angle() == 67.5
    assert beams.pierce_angle(degrees=False) == pytest.approx(np.radians(67.5))
    with pytest.raises(ValueError):
        beams.beam_divergence_gaussian(1.0, 1.0, "fwhm")


def test_aperture_lens_against_paraxial_ray():
    # trace a paraxial ray (nonrelativistic) through a thin field step E1 -> E2 across the aperture:
    # r'' + phi'/(2 phi) r' + phi''/(4 phi) r = 0
    voltage, e1, e2, w = 30e3, 2e5, 1e6, 1e-6

    def phi(z):
        return voltage + e1 * z + (e2 - e1) * 0.5 * (z + w * np.log(np.cosh(z / w)))

    def dphi(z):
        return e1 + (e2 - e1) * 0.5 * (1 + np.tanh(z / w))

    def d2phi(z):
        return (e2 - e1) * 0.5 / w / np.cosh(z / w) ** 2

    def f(z, y):
        return np.array([y[1], -dphi(z) / (2 * phi(z)) * y[1] - d2phi(z) / (4 * phi(z)) * y[0]])

    z, h, y = -30 * w, w / 20, np.array([1e-3, 0.0])
    while z < 30 * w - 1e-12:
        k1 = f(z, y)
        k2 = f(z + h / 2, y + h / 2 * k1)
        k3 = f(z + h / 2, y + h / 2 * k2)
        k4 = f(z + h, y + h * k3)
        y, z = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), z + h
    focal = -y[0] / y[1]
    assert focal == pytest.approx(beams.aperture_lens_focal_length(voltage, e1, e2), rel=2e-2)
    assert beams.aperture_lens_focal_length(voltage, 1.0, 5e5) > 0 > beams.aperture_lens_focal_length(voltage, 5e5, 1.0)
    assert np.isinf(beams.aperture_lens_focal_length(voltage, 1e5, 1e5))


def test_attenuation_and_compensation_time():
    assert beams.neutral_beam_attenuation(1e19, 1e-20, length=5.0) == pytest.approx(np.exp(-0.5))
    assert beams.stripping_fraction(1e19, 1e-20, length=5.0) == pytest.approx(1 - np.exp(-0.5))
    # a parabolic density profile: integral of n0 (1 - (l/L)^2) over [0, L] is 2/3 n0 L
    x = np.linspace(0, 2.0, 2001)
    density = 1e19 * (1 - (x / 2.0) ** 2)
    assert beams.neutral_beam_attenuation(density, 1e-20, positions=x) == pytest.approx(
        np.exp(-1e19 * 1e-20 * 2 / 3 * 2.0), rel=1e-6
    )
    # a cross section varying along the path, and a batch of profiles (axis 0)
    sigma = 1e-20 * (1 + x)
    out = beams.neutral_beam_attenuation(density, sigma, positions=x)
    assert out == pytest.approx(np.exp(-np.trapezoid(density * sigma, x)))
    batch = np.stack([density, 2 * density])
    out = beams.neutral_beam_attenuation(batch.T, 1e-20, positions=x, axis=0)
    np.testing.assert_allclose(out, np.exp(-np.array([1.0, 2.0]) * 1e19 * 1e-20 * 4 / 3), rtol=1e-6)
    with pytest.raises(ValueError):
        beams.neutral_beam_attenuation(1e19, 1e-20)
    # tau = 1/(n sigma v): the neutralising density grows as n_b n_g sigma v t
    n_g, sigma, v = 1e18, 2e-20, 1e6
    assert beams.space_charge_compensation_time(n_g, sigma, v) == pytest.approx(5e-5)
    assert beams.space_charge_compensation_time(np.array([n_g, 2 * n_g]), sigma, v).shape == (2,)


def _field_energy_factor(density, r):
    """u = 16 pi eps0 (W - W_u)/lambda^2 for a radial profile in a pipe of radius r[-1].

    With f(r) the enclosed charge fraction, 4 pi eps0 W/lambda^2 = int f^2 / r dr, and a uniform beam
    of the same rms radius has 1/4 + ln(r_pipe/r_b).
    """
    shell = density * r
    f = np.concatenate([[0.0], np.cumsum(0.5 * (shell[1:] + shell[:-1]) * np.diff(r))])
    f /= f[-1]
    x2 = np.trapezoid(density * r**3, r) / np.trapezoid(shell, r) / 2
    rb = 2 * np.sqrt(x2)
    integrand = np.zeros_like(r)
    integrand[1:] = f[1:] ** 2 / r[1:]
    return 4 * (np.trapezoid(integrand, r) - (0.25 + np.log(r[-1] / rb)))


def test_emittance_growth_space_charge():
    # a KV beam (u = 0) does not change its emittance, whatever its space charge
    assert beams.emittance_growth_space_charge(0.0, 0.3) == 1.0
    assert beams.emittance_growth_space_charge(0.15, 1.0) == 1.0  # no space charge: no growth
    # the closed form, and it is symmetric: u_i = u_f gives 1
    assert beams.emittance_growth_space_charge(0.1, 0.5) == pytest.approx(np.sqrt(1 + 0.5 * 3 * 0.1))
    assert beams.emittance_growth_space_charge(0.1, 0.5, field_energy_final=0.1) == 1.0
    assert beams.emittance_growth_space_charge(0.0, 0.5, field_energy_final=0.1) < 1.0
    assert beams.emittance_growth_space_charge([0.1, 0.2], 0.5).shape == (2,)
    with pytest.raises(ValueError):
        beams.emittance_growth_space_charge(0.1, 0.0)
    with pytest.raises(ValueError):
        beams.emittance_growth_space_charge(0.0, 0.1, field_energy_final=1.0)


def test_field_energy_factor_gaussian_and_uniform():
    # numerical u = 16 pi eps0 (W - W_u)/lambda^2 of a Gaussian in a pipe: 2 gamma_E - 1 (derived from
    # int (1 - e^-u)^2 du / u = ln X + gamma - ln 2 in the large pipe limit)
    r = np.linspace(0.0, 400.0, 400001)
    gauss = np.exp(-(r**2) / 2)
    assert _field_energy_factor(gauss, r) == pytest.approx(2 * np.euler_gamma - 1, abs=2e-3)
    # a uniform (KV) profile has u = 0
    uniform = np.where(r < 2.0, 1.0, 0.0)
    assert _field_energy_factor(uniform, r) == pytest.approx(0.0, abs=2e-3)
    # the Gaussian-to-uniform growth then follows from the formula
    u = _field_energy_factor(gauss, r)
    assert beams.emittance_growth_space_charge(u, 0.2) == pytest.approx(
        np.sqrt(1 + 12 * (2 * np.euler_gamma - 1)), rel=1e-2
    )
