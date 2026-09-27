"""struphy_plots.theory.parameters: NRL Plasma Formulary values, identities, and Struphy's units."""

from types import SimpleNamespace

import numpy as np
import pytest

from struphy_plots.theory import parameters as par

CM = 1e-2  # m
PER_CM3 = 1e6  # m⁻³
GAUSS = 1e-4  # T


def test_constants():
    assert par.ev_to_kelvin == pytest.approx(11604.518, rel=1e-7)
    assert par.ev_to_joule == par.elementary_charge
    assert par.vacuum_permeability * par.vacuum_permittivity * par.speed_of_light**2 == pytest.approx(1, rel=1e-9)
    assert par.proton_mass / par.electron_mass == pytest.approx(1836.15267, rel=1e-8)


@pytest.mark.parametrize("n_cm3, t_ev", [(1.0, 1.0), (1e14, 1e3), (3e12, 20.0)])
def test_nrl_formulary_values(n_cm3, t_ev):
    n = n_cm3 * PER_CM3
    mu, z, b_gauss = 2.0, 1.0, 3e4
    m_i = mu * par.proton_mass
    rtol = 3e-3  # the formulary gives three digits
    assert par.plasma_frequency(n) / (2 * np.pi) == pytest.approx(8.98e3 * np.sqrt(n_cm3), rel=rtol)
    assert par.plasma_frequency(n, m_i) / (2 * np.pi) == pytest.approx(2.10e2 * np.sqrt(n_cm3 / mu), rel=rtol)
    assert par.cyclotron_frequency(1.0) / (2 * np.pi) == pytest.approx(28.0e9, rel=rtol)
    assert par.cyclotron_frequency(1.0, m_i) / (2 * np.pi) == pytest.approx(1.52e7 / mu, rel=rtol)
    assert par.debye_length(n, t_ev) == pytest.approx(7.43e2 * np.sqrt(t_ev / n_cm3) * CM, rel=rtol)
    assert par.thermal_speed(t_ev) == pytest.approx(4.19e7 * np.sqrt(t_ev) * CM, rel=rtol)
    assert par.thermal_speed(t_ev, m_i) == pytest.approx(9.79e5 * np.sqrt(t_ev / mu) * CM, rel=rtol)
    b = b_gauss * GAUSS
    assert par.larmor_radius(b, temperature=t_ev) == pytest.approx(2.38 * np.sqrt(t_ev) / b_gauss * CM, rel=rtol)
    assert par.larmor_radius(b, temperature=t_ev, mass=m_i) == pytest.approx(
        1.02e2 * np.sqrt(mu * t_ev) / (z * b_gauss) * CM, rel=rtol)
    assert par.inertial_length(n) == pytest.approx(5.31e5 / np.sqrt(n_cm3) * CM, rel=rtol)
    assert par.inertial_length(n, m_i) == pytest.approx(2.28e7 * np.sqrt(mu / n_cm3) / z * CM, rel=rtol)
    assert par.alfven_speed(b, n, mass_number=mu) == pytest.approx(2.18e11 * b_gauss / np.sqrt(mu * n_cm3) * CM, rel=rtol)
    assert par.plasma_parameter(n, t_ev) == pytest.approx(1.72e9 * t_ev**1.5 / np.sqrt(n_cm3), rel=rtol)
    assert par.plasma_beta(n, t_ev, b) == pytest.approx(4.03e-11 * n_cm3 * t_ev / b_gauss**2, rel=rtol)
    assert par.sound_speed(t_ev, mass_number=mu, electron_gamma=5 / 3) == pytest.approx(
        9.79e5 * np.sqrt(5 / 3 * t_ev / mu) * CM, rel=rtol)


def test_quoted_reference_values():
    # deuterium, 1 T, 1e20 m⁻³: v_A ≈ 1.54e6 m/s
    assert par.alfven_speed(1.0, 1e20, mass_number=2) == pytest.approx(1.54e6, rel=2e-3)
    assert par.cyclotron_frequency(1.0) / (2 * np.pi) == pytest.approx(2.8e10, rel=1e-3)
    assert par.alfven_speed(1.0, 1e20, relativistic=True) < par.alfven_speed(1.0, 1e20)
    v_a = par.alfven_speed(1.0, 1e14)
    assert par.alfven_speed(1.0, 1e14, relativistic=True) == pytest.approx(v_a / np.sqrt(1 + (v_a / par.speed_of_light) ** 2))
    assert par.alfven_speed(1.0, 1e10, relativistic=True) == pytest.approx(par.speed_of_light, rel=1e-3)


def test_identities():
    n, t, b = np.array([1e18, 1e20]), np.array([[10.0], [1e4]]), 2.5
    # ω_p λ_D = v_th with v_th = √(T/m)
    np.testing.assert_allclose(par.plasma_frequency(n) * par.debye_length(n, t), par.thermal_speed(t) * np.ones(2))
    # d_i = v_A / Ω_i for one ion species (mass A m_p)
    m_i = 2 * par.proton_mass
    d_i = par.inertial_length(n, m_i)
    np.testing.assert_allclose(d_i, par.alfven_speed(b, n, mass_number=2) / par.cyclotron_frequency(b, m_i), rtol=1e-12)
    # thermal-speed conventions
    assert par.thermal_speed(1.0, convention="sqrt(2T/m)") == pytest.approx(np.sqrt(2) * par.thermal_speed(1.0))
    assert par.thermal_speed(1.0, convention="mean") == pytest.approx(np.sqrt(8 / np.pi) * par.thermal_speed(1.0))
    # Larmor radius from a speed; cyclotron sign follows the charge
    assert par.larmor_radius(b, perpendicular_speed=1e5) == pytest.approx(1e5 / par.cyclotron_frequency(b))
    assert par.cyclotron_frequency(-b, charge=-par.elementary_charge) == -par.cyclotron_frequency(b)
    # hybrid frequencies
    omega_ce, omega_ci = par.cyclotron_frequency(b), par.cyclotron_frequency(b, par.proton_mass)
    assert par.upper_hybrid_frequency(1e19, b) ** 2 == pytest.approx(par.plasma_frequency(1e19) ** 2 + omega_ce**2)
    assert par.lower_hybrid_frequency(1e24, b) == pytest.approx(np.sqrt(omega_ce * omega_ci), rel=1e-3)  # dense
    assert par.lower_hybrid_frequency(1e10, b) == pytest.approx(omega_ci, rel=1e-3)  # tenuous: → Ω_i
    # beta is the pressure ratio
    assert par.plasma_beta(1e20, 1e3, 1.0) == pytest.approx(1e20 * 1e3 * par.elementary_charge / (1 / (2 * par.vacuum_permeability)))


def test_broadcasting_and_errors():
    assert par.debye_length(np.ones((2, 1)) * 1e19, np.ones(3)).shape == (2, 3)
    assert par.alfven_speed(np.ones(4), 1e20).shape == (4,)
    assert np.ndim(par.plasma_frequency(1e19)) == 0
    with pytest.raises(ValueError, match="convention"):
        par.thermal_speed(1.0, convention="rms")
    with pytest.raises(ValueError, match="exactly one"):
        par.larmor_radius(1.0)
    with pytest.raises(ValueError, match="exactly one"):
        par.larmor_radius(1.0, temperature=1.0, perpendicular_speed=1.0)
    with pytest.raises(ValueError, match="velocity_scale"):
        par.struphy_units(velocity_scale="sound")
    with pytest.raises(ValueError, match="mass_number"):
        par.struphy_units(velocity_scale="alfvén")
    with pytest.raises(ValueError, match="charge_number"):
        par.struphy_units(velocity_scale="cyclotron", mass_number=1)
    with pytest.raises(ValueError, match="kBT"):
        par.struphy_units(velocity_scale="thermal", mass_number=1)


def test_struphy_units_definitions():
    units = par.struphy_units(x=2.0, B=3.0, n=0.5, velocity_scale="alfvén", mass_number=2)
    assert units["n"] == 0.5e20
    assert units["v"] == pytest.approx(par.alfven_speed(3.0, 0.5e20, mass_number=2))
    assert units["p"] == pytest.approx(3.0**2 / par.vacuum_permeability)  # B²/μ₀ for "alfvén"
    assert units["t"] == pytest.approx(2.0 / units["v"]) and units["frequency"] == pytest.approx(1 / units["t"])
    cyclotron = par.struphy_units(x=0.1, B=2.0, velocity_scale="cyclotron", mass_number=1, charge_number=1)
    assert cyclotron["frequency"] == pytest.approx(par.cyclotron_frequency(2.0, par.proton_mass))
    thermal = par.struphy_units(kBT=2.0, velocity_scale="thermal", mass_number=4)
    assert thermal["v"] == pytest.approx(par.thermal_speed(2e3, 4 * par.proton_mass))
    light = par.struphy_units()
    assert light["v"] == par.speed_of_light and light["p"] is None
    assert par.struphy_units(velocity_scale=None)["v"] == 1.0
    assert par.struphy_units(velocity_scale="alfven", mass_number=1) == par.struphy_units(velocity_scale="alfvén", mass_number=1)
    assert set(light) == set(par.STRUPHY_UNIT_SYMBOLS)
    # equation parameters: α = ω_p/Ω_c, ε = 1/(Ω_c t), κ = ω_p t
    eq = par.struphy_equation_parameters(units, charge_number=1, mass_number=2)
    assert eq["alpha"] * eq["epsilon"] == pytest.approx(eq["kappa"] * eq["epsilon"] ** 2)


@pytest.mark.parametrize("scale, base, bulk", [
    ("light", dict(x=1.0, B=1.0, n=1.0), dict()),
    ("light", dict(x=0.3, B=2.0, n=0.1), dict(A_bulk=1, Z_bulk=-1)),
    ("alfvén", dict(x=2.0, B=3.0, n=0.5), dict(A_bulk=2, Z_bulk=1)),
    ("cyclotron", dict(x=0.01, B=0.5, n=4.0), dict(A_bulk=4, Z_bulk=2)),
    ("thermal", dict(x=1.5, B=1.0, n=1.0, kBT=10.0), dict(A_bulk=1, Z_bulk=1)),
])
def test_struphy_units_match_struphy(scale, base, bulk):
    physics = pytest.importorskip("struphy.physics.physics")
    options = pytest.importorskip("struphy.io.options")
    species = pytest.importorskip("struphy.models.species")
    reference = physics.Units(options.BaseUnits(**base))
    reference.derive_units(velocity_scale=scale, **bulk)
    ours = par.struphy_units(**base, velocity_scale=scale, mass_number=bulk.get("A_bulk"),
                             charge_number=bulk.get("Z_bulk"))
    for key in ("x", "B", "n", "kBT", "v", "t", "p", "rho", "j"):
        expected = getattr(reference, key)
        if expected is None:
            assert ours[key] is None, key
        else:
            assert ours[key] == pytest.approx(float(expected), rel=1e-14), key
    # the constants are Struphy's
    con = physics.ConstantsOfNature()
    assert (con.e, con.mH, con.mu0, con.eps0, con.c) == (par.elementary_charge, par.proton_mass,
                                                      par.vacuum_permeability, par.vacuum_permittivity,
                                                      par.speed_of_light)
    for z, a in [(1, 1), (-1, 1 / 1836), (2, 4)]:
        eq = species.Species.EquationParameters(SimpleNamespace(charge_number=z, mass_number=a), units=reference)
        mine = par.struphy_equation_parameters(ours, charge_number=z, mass_number=a)
        for key in ("alpha", "epsilon", "kappa"):
            assert mine[key] == pytest.approx(float(getattr(eq, key)), rel=1e-13), key
