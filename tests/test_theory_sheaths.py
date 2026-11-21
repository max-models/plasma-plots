"""plasma_plots.theory.sheaths: textbook values and independent identities and limits."""

import math

import numpy as np
import pytest

from plasma_plots.theory import parameters as par
from plasma_plots.theory import sheaths as sh

E = par.elementary_charge
EPS0 = par.vacuum_permittivity
MP = par.proton_mass
ME = par.electron_mass


# ---------------------------------------------------------------------------------------------
# Sheaths
# ---------------------------------------------------------------------------------------------
def test_bohm_speed():
    assert sh.bohm_speed(5.0) == pytest.approx(math.sqrt(5.0 * E / MP), rel=1e-12)
    assert sh.bohm_speed(5.0, mass_number=40) == pytest.approx(
        sh.bohm_speed(5.0) / math.sqrt(40), rel=1e-12
    )
    # warm isothermal ions: (T_e + T_i)/m
    assert sh.bohm_speed(4.0, 1.0) == pytest.approx(math.sqrt(5.0 * E / MP), rel=1e-12)
    assert isinstance(sh.bohm_speed(5.0), float)


def test_bohm_current_density():
    n, te = 1e18, 5.0
    assert sh.bohm_current_density(n, te) == pytest.approx(
        0.6065 * E * n * sh.bohm_speed(te), rel=1e-3
    )
    assert sh.ion_saturation_current(n, te, 1e-6) == pytest.approx(
        1e-6 * sh.bohm_current_density(n, te), rel=1e-12
    )
    # Z e n √(Z T_e/m): twice the charge and √2 the speed
    assert sh.bohm_current_density(n, te, charge_number=2) == pytest.approx(
        2 * math.sqrt(2) * sh.bohm_current_density(n, te), rel=1e-12
    )


def test_floating_potential_textbook_values():
    # Lieberman and Lichtenberg: V_f = -(T_e/2) ln(M/(2 pi m)): 2.84 T_e for hydrogen, 4.68 T_e for argon
    assert sh.floating_potential(1.0) == pytest.approx(-2.84, abs=0.005)
    assert sh.floating_potential(1.0, mass_number=40) == pytest.approx(-4.68, abs=0.005)
    # proportional to T_e
    assert sh.floating_potential(7.0) == pytest.approx(
        7 * sh.floating_potential(1.0), rel=1e-12
    )


def test_floating_potential_balances_the_currents():
    """Solve ion current = net electron current by bisection instead of using the closed form."""
    te, delta, mass_number = 3.0, 0.6, 4.0
    ion_flux = math.sqrt(te * E / (mass_number * MP))
    electron_flux = lambda v: (
        (1 - delta) * math.sqrt(te * E / (2 * math.pi * ME)) * math.exp(v / te)
    )  # noqa: E731
    lo, hi = -100.0, 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if electron_flux(mid) < ion_flux else (lo, mid)
    assert sh.floating_potential(
        te, mass_number, secondary_emission=delta
    ) == pytest.approx(lo, rel=1e-9)
    # emission makes the wall less negative; electron mass enters through the log
    assert sh.floating_potential(te, secondary_emission=0.9) > sh.floating_potential(te)
    assert sh.floating_potential(1.0, electron_mass=4 * ME) == pytest.approx(
        sh.floating_potential(1.0) + 0.5 * math.log(4), rel=1e-12
    )
    # heavier ions: mass_number is a mass ratio
    assert sh.floating_potential(1.0, mass_number=2) == pytest.approx(
        sh.floating_potential(1.0) - 0.5 * math.log(2), rel=1e-12
    )


@pytest.mark.parametrize("delta", [1.0, 1.5, -0.1])
def test_floating_potential_rejects_bad_emission(delta):
    with pytest.raises(ValueError):
        sh.floating_potential(1.0, secondary_emission=delta)


def test_child_langmuir_sheath_width_from_the_space_charge_law():
    """The sheath carries e n u_B; invert the planar child_langmuir law for its width."""
    n, te, v0 = 1e16, 3.0, 100.0
    j = E * n * sh.bohm_speed(te)
    d = 1e-3
    width = d * math.sqrt(sh.child_langmuir(v0, d, mass=MP) / j)
    assert sh.sheath_width(n, te, v0) == pytest.approx(width, rel=1e-10)


def test_matrix_sheath_width_from_poisson():
    n, te, v0 = 1e16, 3.0, 100.0
    width = sh.sheath_width(n, te, v0, model="matrix")
    assert E * n * width**2 / (2 * EPS0) == pytest.approx(v0, rel=1e-10)
    assert sh.sheath_width(n, te, -v0, model="matrix") == width
    # the Child-Langmuir sheath is wider when e V0 >> T_e
    assert sh.sheath_width(n, te, v0) > width
    with pytest.raises(ValueError):
        sh.sheath_width(n, te, v0, model="other")


# ---------------------------------------------------------------------------------------------
# Child-Langmuir and Langmuir-Blodgett
# ---------------------------------------------------------------------------------------------
def test_child_langmuir_matches_the_picnic_value():
    m = 1.00794 * MP
    v, d = 1e4, 1e-2
    picnic = 4 / 9 * EPS0 * math.sqrt(2 * E / m) * v**1.5 / d**2
    assert sh.child_langmuir(v, d, mass=m) == pytest.approx(picnic, rel=1e-12)


def test_child_langmuir_textbook_and_scaling():
    # electrons: 2.33e-6 A V^-3/2 (Child's constant)
    assert sh.child_langmuir(1.0, 1.0) == pytest.approx(2.334e-6, rel=1e-3)
    assert sh.child_langmuir(4.0, 1.0) == pytest.approx(
        8 * sh.child_langmuir(1.0, 1.0), rel=1e-12
    )
    assert sh.child_langmuir(1.0, 0.1) == pytest.approx(
        100 * sh.child_langmuir(1.0, 1.0), rel=1e-12
    )
    assert sh.child_langmuir(1.0, 1.0, mass=4 * ME) == pytest.approx(
        sh.child_langmuir(1.0, 1.0) / 2, rel=1e-12
    )
    assert sh.child_langmuir(1.0, 1.0, charge=-4 * E) == pytest.approx(
        2 * sh.child_langmuir(1.0, 1.0), rel=1e-12
    )
    v = np.array([1.0, 4.0])
    assert sh.child_langmuir(v, 1.0).shape == (2,)


@pytest.mark.parametrize("w0", [0.3, 1.0, 5.0])
def test_child_langmuir_injected_beam_satisfies_poisson(w0):
    """Integrate Poisson's equation across the gap with RK4: the potential must reach V at d.

    With ψ = φ + m v₀²/(2q) and v = √(2q/m) √ψ, the charge density is j/v and
    ψ'' = a ψ^(-1/2), a = j / (ε₀ √(2q/m)), ψ(0) = w₀² and ψ'(0) = 0 (space-charge limited).
    """
    m, q, v, d = 1.0e-27, 1.6e-19, 50.0, 2e-3
    scale = math.sqrt(2 * q / m)
    j = sh.child_langmuir(
        v, d, mass=m, charge=q, initial_velocity=w0 * math.sqrt(v) * scale
    )
    a = j / (EPS0 * scale)
    w0 = w0 * math.sqrt(v)

    def rhs(y):
        return np.array([y[1], a / math.sqrt(y[0])])

    y = np.array([w0**2, 0.0])
    steps = 4000
    h = d / steps
    for _ in range(steps):
        k1 = rhs(y)
        k2 = rhs(y + h / 2 * k1)
        k3 = rhs(y + h / 2 * k2)
        k4 = rhs(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert y[0] - w0**2 == pytest.approx(v, rel=1e-6)


def test_child_langmuir_injection_limits():
    m, q, v, d = ME, E, 100.0, 1e-3
    scale = math.sqrt(2 * q / m)
    # a fast beam makes space charge negligible: J = 2 eps0 v0 V / d²
    v0 = 1e3 * math.sqrt(v) * scale
    assert sh.child_langmuir(v, d, initial_velocity=v0) == pytest.approx(
        2 * EPS0 * v0 * v / d**2, rel=2e-3
    )
    # injection always increases the current
    assert sh.child_langmuir(v, d, initial_velocity=1e6) > sh.child_langmuir(v, d)
    assert sh.child_langmuir(v, d, initial_velocity=0.0) == sh.child_langmuir(v, d)


def lb_phi(gamma, geometry):
    """φ(γ) with the Langmuir normalisation C = 1, from the series."""
    if geometry == "cylindrical":
        return np.exp(2 * gamma / 3) * sh._lb_function(gamma, sh._LB_CYLINDRICAL) ** (
            4 / 3
        )
    return sh._lb_function(gamma, sh._LB_SPHERICAL) ** (4 / 3)


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
@pytest.mark.parametrize("gamma", [0.2, 1.0, 2.5, 4.0])
def test_langmuir_blodgett_series_solves_poisson(geometry, gamma):
    """The series gives φ with φ_γγ = k r φ^(-1/2) (cylinders) or φ_γγ + φ_γ = k φ^(-1/2) (spheres)."""
    k, h = 4 / 9, 1e-3
    phi = lb_phi(np.array([gamma - h, gamma, gamma + h]), geometry)
    d1 = (phi[2] - phi[0]) / (2 * h)
    d2 = (phi[2] - 2 * phi[1] + phi[0]) / h**2
    if geometry == "cylindrical":
        lhs, rhs = d2, k * math.exp(gamma) * phi[1] ** -0.5
    else:
        lhs, rhs = d2 + d1, k * phi[1] ** -0.5
    assert lhs == pytest.approx(rhs, rel=1e-5)


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_langmuir_blodgett_tends_to_child_langmuir(geometry):
    v, r_c, eps = 500.0, 0.02, 1e-4
    r_a = r_c * (1 + eps)
    planar = sh.child_langmuir(v, r_a - r_c, mass=MP)
    assert sh.langmuir_blodgett(v, r_a, r_c, geometry, mass=MP) == pytest.approx(
        planar, rel=1e-3
    )
    # emitter outside: converging flow, same limit
    r_a = r_c * (1 - eps)
    planar = sh.child_langmuir(v, r_c - r_a, mass=MP)
    assert sh.langmuir_blodgett(v, r_a, r_c, geometry, mass=MP) == pytest.approx(
        planar, rel=1e-3
    )


@pytest.mark.parametrize("geometry", ["cylindrical", "spherical"])
def test_langmuir_blodgett_total_current_and_scaling(geometry):
    v, r_a, r_c = 800.0, 4e-3, 1e-3
    j = sh.langmuir_blodgett(v, r_a, r_c, geometry)
    i = sh.langmuir_blodgett(v, r_a, r_c, geometry, total=True)
    area = 2 * math.pi * r_a if geometry == "cylindrical" else 4 * math.pi * r_a**2
    assert i == pytest.approx(j * area, rel=1e-12)
    assert sh.langmuir_blodgett(4 * v, r_a, r_c, geometry) == pytest.approx(
        8 * j, rel=1e-12
    )
    # any mass and charge
    assert sh.langmuir_blodgett(v, r_a, r_c, geometry, mass=4 * ME) == pytest.approx(
        j / 2, rel=1e-12
    )


def test_langmuir_blodgett_series_coefficients():
    # the classic first coefficients
    assert sh._LB_CYLINDRICAL[:4] == pytest.approx(
        [1.0, -2 / 5, 11 / 120, -47 / 3300], rel=1e-14
    )
    assert sh._LB_SPHERICAL[:4] == pytest.approx(
        [1.0, -3 / 10, 3 / 40, -63 / 4400], rel=1e-14
    )
    # the cylindrical current of an inner emitter is below the planar one for the same gap
    v, r_c, r_a = 500.0, 1e-3, 5e-3
    assert sh.langmuir_blodgett(v, r_a, r_c) < sh.child_langmuir(v, r_a - r_c)


def test_langmuir_blodgett_errors():
    with pytest.raises(ValueError):
        sh.langmuir_blodgett(1.0, 1.0, 0.5, geometry="planar")
    with pytest.raises(ValueError):
        sh.langmuir_blodgett(1.0, 1e3, 1.0)


# ---------------------------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------------------------
def test_erfcx_matches_math_and_the_asymptote():
    x = np.array([0.0, 0.5, 2.0, 10.0, 24.9])
    assert sh._erfcx(x) == pytest.approx(
        [math.exp(v * v) * math.erfc(v) for v in x], rel=1e-10
    )
    assert sh._erfcx(25.0) == pytest.approx(sh._erfcx(24.999999), rel=1e-5)
    assert np.isfinite(sh._erfcx(1e3))


def test_oml_random_current_and_repelled_branch():
    n, te, r, length = 1e14, 2.0, 0.5e-3, 1e-2
    area = 2 * math.pi * r * length
    random = area * n * E * math.sqrt(te * E / (2 * math.pi * ME))
    assert sh.oml_current(n, te, 0.0, r, length) == pytest.approx(random, rel=1e-12)
    # electrons repelled by a probe 3 T_e below the plasma potential
    assert sh.oml_current(n, te, -3 * te, r, length) == pytest.approx(
        random * math.exp(-3), rel=1e-12
    )
    # sphere
    assert sh.oml_current(n, te, 0.0, r, geometry="spherical") == pytest.approx(
        random * (4 * math.pi * r**2) / area, rel=1e-12
    )


def test_oml_attracted_branches():
    n, te, r, length = 1e14, 2.0, 0.5e-3, 1e-2
    random = sh.oml_current(n, te, 0.0, r, length)
    sphere0 = sh.oml_current(n, te, 0.0, r, geometry="spherical")
    # sphere: 1 + χ
    assert sh.oml_current(n, te, 5 * te, r, geometry="spherical") == pytest.approx(
        6 * sphere0, rel=1e-12
    )
    # cylinder: 2√(χ/π) at large χ, 1 + χ at small χ, continuous across χ = 0
    assert sh.oml_current(n, te, 1e4 * te, r, length) / random == pytest.approx(
        2 * math.sqrt(1e4 / math.pi), rel=1e-3
    )
    assert sh.oml_current(n, te, 1e-3 * te, r, length) / random == pytest.approx(
        1.001, rel=1e-4
    )
    assert sh.oml_current(n, te, 1e-12, r, length) == pytest.approx(random, rel=1e-9)
    # exact cylinder value at χ = 4: exp(4) erfc(2) + 4/√π
    expected = math.exp(4) * math.erfc(2) + 4 / math.sqrt(math.pi)
    assert sh.oml_current(n, te, 4 * te, r, length) / random == pytest.approx(
        expected, rel=1e-10
    )


def test_oml_ions_and_errors():
    n, te, r, length = 1e14, 2.0, 0.5e-3, 1e-2
    electrons = sh.oml_current(n, te, 0.0, r, length)
    ions = sh.oml_current(n, te, 0.0, r, length, mass=MP, charge=E)
    assert ions / electrons == pytest.approx(math.sqrt(ME / MP), rel=1e-12)
    # an attracting negative probe collects ions
    assert sh.oml_current(n, te, -10.0, r, length, mass=MP, charge=E) > ions
    with pytest.raises(ValueError):
        sh.oml_current(n, te, 0.0, r)
    with pytest.raises(ValueError):
        sh.oml_current(n, te, 0.0, r, length, geometry="planar")


# ---------------------------------------------------------------------------------------------
# Transport and breakdown
# ---------------------------------------------------------------------------------------------
def test_ambipolar_diffusion_limits():
    mu_i, te, ti = 2e-4, 3.0, 0.03
    # D_i (1 + T_e/T_i) with D_i = μ_i T_i
    assert sh.ambipolar_diffusion(mu_i, te, ti) == pytest.approx(
        mu_i * ti * (1 + te / ti), rel=1e-12
    )
    # general form reduces to it for μ_e >> μ_i and to the electron-limited value for μ_e << μ_i
    assert sh.ambipolar_diffusion(
        mu_i, te, ti, electron_mobility=1e12
    ) == pytest.approx(sh.ambipolar_diffusion(mu_i, te, ti), rel=1e-6)
    d_e = 1e-2 * te
    mu_e = 1e-2
    expected = (mu_i * d_e + mu_e * mu_i * ti) / (mu_i + mu_e)
    assert sh.ambipolar_diffusion(
        mu_i, te, ti, electron_mobility=mu_e
    ) == pytest.approx(expected, rel=1e-12)


def test_bohm_diffusion():
    assert sh.bohm_diffusion(10.0, 0.01) == pytest.approx(62.5)
    assert sh.bohm_diffusion(10.0, -0.01) == pytest.approx(62.5)
    assert sh.bohm_diffusion(np.array([1.0, 2.0]), 1.0) == pytest.approx(
        [1 / 16, 1 / 8]
    )


def test_paschen_minimum_and_branches():
    a, b = sh.PASCHEN_GASES["air"]
    gamma = 0.01
    pd = np.logspace(-2.5, 2, 20001)
    v = sh.paschen_breakdown(pd, 1.0, secondary_emission=gamma)
    k = np.nanargmin(v)
    # analytic minimum: V = e B/A ln(1 + 1/γ) at p d = e ln(1 + 1/γ)/A
    pd_min = math.e * math.log(1 + 1 / gamma) / a
    v_min = math.e * b / a * math.log(1 + 1 / gamma)
    assert pd[k] == pytest.approx(pd_min, rel=2e-3)
    assert v[k] == pytest.approx(v_min, rel=1e-6)
    # air: measured minimum 327 V at 0.567 Torr cm (Naidu, High Voltage Engineering, p. 27)
    assert v_min == pytest.approx(327.0, rel=0.08)
    assert pd_min / 133.322368 * 1e2 == pytest.approx(0.567, rel=0.5)
    # the same value in 1/(cm Torr) units: B p d with p d = 1 Torr cm
    assert sh.paschen_breakdown(133.322368, 1e-2) == pytest.approx(
        365.0 / (math.log(15.0) - math.log(math.log(101.0))), rel=1e-9
    )
    # left of the asymptote there is no breakdown
    assert math.isnan(sh.paschen_breakdown(1.0, 1e-3))
    assert sh.paschen_breakdown(1e5, 1e-3, gas="Ar") > 0
    assert sh.paschen_breakdown(1e5, 1e-3, constants=(a, b)) == sh.paschen_breakdown(
        1e5, 1e-3
    )
    with pytest.raises(ValueError):
        sh.paschen_breakdown(1e5, 1e-3, gas="xenon")


# measured minimum sparking potential (V) and p d (Torr cm), Naidu p. 27, with the largest
# relative deviation of the model (γ = 0.01) in the voltage and in p d
_MEASURED_MINIMA = {
    "air": (327.0, 0.567, 0.08, 0.5),
    "N2": (251.0, 0.67, 0.45, 0.6),
    "H2": (273.0, 1.15, 0.25, 1.2),
    "He": (156.0, 4.0, 0.1, 0.1),
    "Ar": (137.0, 0.9, 0.4, 0.2),
}


@pytest.mark.parametrize("gas", list(_MEASURED_MINIMA))
def test_paschen_minima_against_measurements(gas):
    v_meas, pd_meas, v_tol, pd_tol = _MEASURED_MINIMA[gas]
    a, b = sh.PASCHEN_GASES[gas]
    torr_cm = 133.322368 * 1e-2
    pd = np.linspace(0.05, 20.0, 40001) * torr_cm
    v = sh.paschen_breakdown(pd, 1.0, gas=gas)
    k = np.nanargmin(v)
    assert v[k] == pytest.approx(v_meas, rel=v_tol)
    assert pd[k] / torr_cm == pytest.approx(pd_meas, rel=pd_tol)


def test_paschen_helium_constant():
    # B = 34 V/(cm Torr) (not 25) for helium
    a, b = sh.PASCHEN_GASES["He"]
    assert b * 133.322368 * 1e-2 == pytest.approx(34.0)
    assert a * 133.322368 * 1e-2 == pytest.approx(3.0)


def test_mean_free_path():
    assert sh.mean_free_path(1e-19, density=1e20) == pytest.approx(0.1, rel=1e-12)
    n = 1.0 / (par.boltzmann_constant * 300.0)
    assert sh.mean_free_path(1e-19, pressure=1.0) == pytest.approx(
        1 / (n * 1e-19), rel=1e-12
    )
    # n ∝ 1/T
    assert sh.mean_free_path(1.0, pressure=1.0, temperature=600.0) == pytest.approx(
        2 * sh.mean_free_path(1.0, pressure=1.0, temperature=300.0), rel=1e-12
    )
    assert sh.mean_free_path(np.array([1e-19, 2e-19]), density=1e20) == pytest.approx(
        [0.1, 0.05]
    )
    with pytest.raises(ValueError):
        sh.mean_free_path(1e-19)
    with pytest.raises(ValueError):
        sh.mean_free_path(1e-19, density=1e20, pressure=1.0)
