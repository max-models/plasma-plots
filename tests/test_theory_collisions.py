"""plasma_plots.theory.collisions: NRL formulary values, limits and the Stix slowing-down identities."""

import math

import numpy as np
import pytest

from plasma_plots.theory import collisions as col
from plasma_plots.theory import parameters as par

PER_CM3 = 1e6


def test_coulomb_logarithm_hand_values():
    n_cm3 = 1e13
    # electron-ion above 10 Z^2 eV: 24 - ln(sqrt(n)/T)
    assert col.coulomb_logarithm(1e19, 100.0) == pytest.approx(24 - math.log(math.sqrt(n_cm3) / 100.0), abs=1e-12)
    # below: 23 - ln(sqrt(n) Z T^-1.5)
    assert col.coulomb_logarithm(1e19, 5.0, charge_number=2) == pytest.approx(
        23 - math.log(math.sqrt(n_cm3) * 2 * 5.0**-1.5), abs=1e-12
    )
    # electron-electron
    t = 100.0
    expected = 23.5 - math.log(math.sqrt(n_cm3) * t**-1.25) - math.sqrt(1e-5 + (math.log(t) - 2) ** 2 / 16)
    assert col.coulomb_logarithm(1e19, t, kind="ee") == pytest.approx(expected, abs=1e-12)
    # typical tokamak values are 15-20
    assert 15 < col.coulomb_logarithm(1e20, 5e3) < 20


def test_coulomb_logarithm_regimes_nearly_continuous_and_broadcast():
    for z in (1.0, 2.0, 6.0):
        t_switch = 10 * z**2
        below = col.coulomb_logarithm(1e20, t_switch * (1 - 1e-9), charge_number=z)
        above = col.coulomb_logarithm(1e20, t_switch, charge_number=z)
        assert abs(above - below) < 0.2
    out = col.coulomb_logarithm(np.array([1e18, 1e19, 1e20])[:, None], np.array([20.0, 200.0]))
    assert out.shape == (3, 2)
    assert np.all(np.diff(out, axis=0) < 0)  # denser: smaller
    assert np.isscalar(col.coulomb_logarithm(1e19, 100.0)) or np.ndim(col.coulomb_logarithm(1e19, 100.0)) == 0
    with pytest.raises(ValueError):
        col.coulomb_logarithm(1e19, 100.0, kind="ii")


@pytest.mark.parametrize("n_cm3, t_ev", [(1e13, 100.0), (1e14, 1e3), (1e10, 30.0)])
def test_collision_frequencies_nrl_prefactors(n_cm3, t_ev):
    lnl, mu, z = 15.0, 2.0, 1.0
    nu = col.collision_frequencies(n_cm3 * PER_CM3, t_ev, mass_number=mu, charge_number=z, ln_lambda=lnl)
    assert nu["electron"] == pytest.approx(2.91e-6 * n_cm3 * lnl * t_ev**-1.5, rel=2e-3)
    assert nu["ion"] == pytest.approx(4.80e-8 * z**4 * mu**-0.5 * n_cm3 * lnl * t_ev**-1.5, rel=2e-3)
    assert nu["electron_ion"] == pytest.approx(nu["electron"])


def test_collision_frequencies_charge_scaling_and_default_lnl():
    nu1 = col.collision_frequencies(1e19, 100.0, ln_lambda=15.0)
    nu3 = col.collision_frequencies(1e19, 100.0, charge_number=3.0, ln_lambda=15.0)
    assert nu3["electron"] == pytest.approx(nu1["electron"])
    assert nu3["electron_ion"] == pytest.approx(3.0 * nu1["electron_ion"])  # Z^2 n_i/n_e = Z
    default = col.collision_frequencies(1e19, 100.0)
    lnl = col.coulomb_logarithm(1e19, 100.0)
    assert default["electron"] == pytest.approx(col.collision_frequencies(1e19, 100.0, ln_lambda=lnl)["electron"])
    # ion rate scales as T_i^-3/2 and m_i^-1/2
    a = col.collision_frequencies(1e19, 100.0, ion_temperature=100.0, ln_lambda=15.0)["ion"]
    b = col.collision_frequencies(1e19, 100.0, ion_temperature=400.0, mass_number=4.0, ln_lambda=15.0)["ion"]
    assert b == pytest.approx(a / 8 / 2)


def test_spitzer_resistivity_known_values():
    eta = col.spitzer_resistivity(1000.0, ln_lambda=1.0)
    assert eta == pytest.approx(1.65e-9, rel=2e-2)
    assert eta == pytest.approx(5.2e-5 / 1000**1.5 * 1e0, rel=2.5e-2)  # NRL: 5.2e-5 Z lnL T^-3/2
    # scalings: T^-3/2, lnL, Z_eff at fixed alpha_0 -> alpha_0(Z) Z
    assert col.spitzer_resistivity(2000.0, ln_lambda=1.0) == pytest.approx(eta / 2**1.5)
    assert col.spitzer_resistivity(1000.0, ln_lambda=17.0) == pytest.approx(17 * eta)
    ratio = col.spitzer_resistivity(1000.0, z_eff=2.0, ln_lambda=1.0) / eta
    assert ratio == pytest.approx(2 * 0.4408 / 0.5129)
    # Z -> infinity: alpha_0 -> 0.2949
    assert col.spitzer_resistivity(1000.0, z_eff=1e6, ln_lambda=1.0) / (1e6 * eta) == pytest.approx(
        0.2949 / 0.5129, rel=1e-4
    )


def test_spitzer_resistivity_from_density_and_errors():
    lnl = col.coulomb_logarithm(1e19, 1000.0)
    assert col.spitzer_resistivity(1000.0, density=1e19) == pytest.approx(
        col.spitzer_resistivity(1000.0, ln_lambda=lnl)
    )
    with pytest.raises(ValueError):
        col.spitzer_resistivity(1000.0)


def test_spitzer_resistivity_matches_momentum_transfer_rate():
    # eta_par = alpha_0 m_e nu_ei / (n_e e^2) for Z = 1
    n, t, lnl = 1e19, 500.0, 16.0
    nu = col.collision_frequencies(n, t, ln_lambda=lnl)["electron_ion"]
    expected = 0.5129 * par.electron_mass * nu / (n * par.elementary_charge**2)
    assert col.spitzer_resistivity(t, ln_lambda=lnl) == pytest.approx(expected, rel=1e-12)


def test_slowing_down_time_spitzer_value():
    n_cm3, t_ev, a, lnl = 1e13, 1000.0, 2.0, 17.0
    tau = col.slowing_down_time(n_cm3 * PER_CM3, t_ev, beam_mass_number=a, ln_lambda=lnl)
    assert tau == pytest.approx(6.3e8 * a * t_ev**1.5 / (n_cm3 * lnl), rel=3e-3)
    # scalings: ~ A_b / Z_b^2 / n_e, T_e^3/2
    assert col.slowing_down_time(2e19, t_ev, a, ln_lambda=lnl) == pytest.approx(tau / 2)
    assert col.slowing_down_time(1e19, t_ev, a, 2.0, ln_lambda=lnl) == pytest.approx(tau / 4)
    assert col.slowing_down_time(1e19, t_ev, 2 * a, ln_lambda=lnl) == pytest.approx(2 * tau)


def test_critical_energy_values_and_scalings():
    # Wesson: E_c = 14.8 T_e A_b [sum n_j Z_j^2/(n_e A_j)]^(2/3)
    assert col.critical_energy(1.0) == pytest.approx(14.8, rel=1e-3)
    assert col.critical_energy(1000.0, 2.0, 2.0) == pytest.approx(14.8 * 1000 * 2 * 0.5 ** (2 / 3), rel=1e-3)
    # hydrogenic species list equals the single-species form
    ions = [(0.5, 1, 2), (0.5, 1, 2)]
    assert col.critical_energy(1e3, 2.0, ions=ions) == pytest.approx(col.critical_energy(1e3, 2.0, 2.0))
    # impurity: carbon 6+ with n_C/n_e = 0.1 (Z_eff-ish) raises the sum: 0.4 D + 0.1 C(6, A=12) -> check by hand
    s = 0.4 / 2 + 0.1 * 36 / 12
    assert col.critical_energy(1e3, 2.0, ions=[(0.4, 1, 2), (0.1, 6, 12)]) == pytest.approx(
        14.8 * 1e3 * 2.0 * s ** (2 / 3), rel=1e-3
    )
    # beam ion with equal speed to the critical one: E_c = 1/2 m_b v_c^2, v_c^3 from the definition
    t, a = 800.0, 3.0
    v_te = math.sqrt(2 * t * par.ev_to_joule / par.electron_mass)
    v_c = (3 * math.sqrt(math.pi) / 4 * par.electron_mass / par.proton_mass * v_te**3) ** (1 / 3)  # hydrogen target
    assert col.critical_energy(t, a) * par.ev_to_joule == pytest.approx(0.5 * a * par.proton_mass * v_c**2, rel=1e-12)


def test_slowing_down_energy_time_roundtrip_and_limits():
    n, t, a, lnl = 1e19, 2000.0, 2.0, 16.0
    e_c = col.critical_energy(t, a, 2.0)
    tau = col.slowing_down_time(n, t, a, ln_lambda=lnl)
    kw = dict(density=n, temperature=t, beam_mass_number=a, ion_mass_number=2.0, ln_lambda=lnl)
    e0 = 100e3
    for e in (80e3, 30e3, e_c, 5e3):
        time = col.slowing_down_time(n, t, a, ln_lambda=lnl, energy=e0, final_energy=e, critical_energy_ev=e_c)
        assert col.beam_stopping_fast_ion(e0, time=time, **kw) == pytest.approx(e, rel=1e-9)
    # thermalisation time: finite, and the energy is 0 afterwards
    t_th = col.slowing_down_time(n, t, a, ln_lambda=lnl, energy=e0, critical_energy_ev=e_c)
    assert col.beam_stopping_fast_ion(e0, time=1.01 * t_th, **kw) == 0.0
    # electron drag only (E_c -> 0): v = v0 exp(-t/tau_s), E = E0 exp(-2t/tau_s)
    kw0 = dict(kw, ion_mass_number=1e12)
    assert col.beam_stopping_fast_ion(e0, time=0.3 * tau, **kw0) == pytest.approx(e0 * math.exp(-0.6), rel=1e-6)
    t_e = col.slowing_down_time(n, t, a, ln_lambda=lnl, energy=e0, final_energy=e0 / 10, critical_energy_ev=1e-9)
    assert t_e == pytest.approx(tau / 2 * math.log(10), rel=1e-6)
    # no ion drag with an infinite critical energy is not what np.inf means: time diverges to tau/3 * ln(1)
    assert col.slowing_down_time(n, t, a, ln_lambda=lnl, energy=e0, critical_energy_ev=np.inf) == 0.0


def test_beam_stopping_rates():
    n, t, a, lnl = 1e19, 1000.0, 2.0, 17.0
    kw = dict(density=n, temperature=t, beam_mass_number=a, ion_mass_number=2.0, ln_lambda=lnl)
    e_c = col.critical_energy(t, a, 2.0)
    tau = col.slowing_down_time(n, t, a, ln_lambda=lnl)
    loss = col.beam_stopping_fast_ion(80e3, **kw)
    assert loss["total"] == pytest.approx(loss["electrons"] + loss["ions"])
    assert loss["electrons"] == pytest.approx(2 * 80e3 / tau)
    assert loss["ions"] / loss["electrons"] == pytest.approx((e_c / 80e3) ** 1.5)
    # equal at the critical energy
    at_c = col.beam_stopping_fast_ion(e_c, **kw)
    assert at_c["ions"] == pytest.approx(at_c["electrons"])
    # rate is the derivative of the energy after t
    dt = 1e-7
    e1 = col.beam_stopping_fast_ion(80e3, time=dt, **kw)
    assert (80e3 - e1) / dt == pytest.approx(loss["total"], rel=1e-4)
    # an array of energies broadcasts
    assert col.beam_stopping_fast_ion(np.array([20e3, 80e3]), **kw)["total"].shape == (2,)
