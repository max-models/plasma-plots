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
    assert col.coulomb_logarithm(1e19, 100.0) == pytest.approx(
        24 - math.log(math.sqrt(n_cm3) / 100.0), abs=1e-12
    )
    # below: 23 - ln(sqrt(n) Z T^-1.5)
    assert col.coulomb_logarithm(1e19, 5.0, charge_number=2) == pytest.approx(
        23 - math.log(math.sqrt(n_cm3) * 2 * 5.0**-1.5), abs=1e-12
    )
    # electron-electron
    t = 100.0
    expected = (
        23.5
        - math.log(math.sqrt(n_cm3) * t**-1.25)
        - math.sqrt(1e-5 + (math.log(t) - 2) ** 2 / 16)
    )
    assert col.coulomb_logarithm(1e19, t, kind="ee") == pytest.approx(
        expected, abs=1e-12
    )
    # typical tokamak values are 15-20
    assert 15 < col.coulomb_logarithm(1e20, 5e3) < 20


def test_coulomb_logarithm_regimes_nearly_continuous_and_broadcast():
    for z in (1.0, 2.0, 6.0):
        t_switch = 10 * z**2
        below = col.coulomb_logarithm(1e20, t_switch * (1 - 1e-9), charge_number=z)
        above = col.coulomb_logarithm(1e20, t_switch, charge_number=z)
        assert abs(above - below) < 0.2
    out = col.coulomb_logarithm(
        np.array([1e18, 1e19, 1e20])[:, None], np.array([20.0, 200.0])
    )
    assert out.shape == (3, 2)
    assert np.all(np.diff(out, axis=0) < 0)  # denser: smaller
    assert (
        np.isscalar(col.coulomb_logarithm(1e19, 100.0))
        or np.ndim(col.coulomb_logarithm(1e19, 100.0)) == 0
    )
    with pytest.raises(ValueError):
        col.coulomb_logarithm(1e19, 100.0, kind="ii")


@pytest.mark.parametrize("n_cm3, t_ev", [(1e13, 100.0), (1e14, 1e3), (1e10, 30.0)])
def test_collision_frequencies_nrl_prefactors(n_cm3, t_ev):
    lnl, mu, z = 15.0, 2.0, 1.0
    nu = col.collision_frequencies(
        n_cm3 * PER_CM3, t_ev, mass_number=mu, charge_number=z, ln_lambda=lnl
    )
    assert nu["electron"] == pytest.approx(2.91e-6 * n_cm3 * lnl * t_ev**-1.5, rel=2e-3)
    assert nu["ion"] == pytest.approx(
        4.80e-8 * z**4 * mu**-0.5 * n_cm3 * lnl * t_ev**-1.5, rel=2e-3
    )
    assert nu["electron_ion"] == pytest.approx(nu["electron"])


def test_collision_frequencies_charge_scaling_and_default_lnl():
    nu1 = col.collision_frequencies(1e19, 100.0, ln_lambda=15.0)
    nu3 = col.collision_frequencies(1e19, 100.0, charge_number=3.0, ln_lambda=15.0)
    assert nu3["electron"] == pytest.approx(nu1["electron"])
    assert nu3["electron_ion"] == pytest.approx(
        3.0 * nu1["electron_ion"]
    )  # Z^2 n_i/n_e = Z
    default = col.collision_frequencies(1e19, 100.0)
    lnl = col.coulomb_logarithm(1e19, 100.0)
    assert default["electron"] == pytest.approx(
        col.collision_frequencies(1e19, 100.0, ln_lambda=lnl)["electron"]
    )
    # ion rate scales as T_i^-3/2 and m_i^-1/2
    a = col.collision_frequencies(1e19, 100.0, ion_temperature=100.0, ln_lambda=15.0)[
        "ion"
    ]
    b = col.collision_frequencies(
        1e19, 100.0, ion_temperature=400.0, mass_number=4.0, ln_lambda=15.0
    )["ion"]
    assert b == pytest.approx(a / 8 / 2)


def test_spitzer_resistivity_known_values():
    eta = col.spitzer_resistivity(1000.0, ln_lambda=1.0)
    assert eta == pytest.approx(1.65e-9, rel=2e-2)
    assert eta == pytest.approx(
        5.2e-5 / 1000**1.5 * 1e0, rel=2.5e-2
    )  # NRL: 5.2e-5 Z lnL T^-3/2
    # scalings: T^-3/2, lnL, Z_eff at fixed alpha_0 -> alpha_0(Z) Z
    assert col.spitzer_resistivity(2000.0, ln_lambda=1.0) == pytest.approx(eta / 2**1.5)
    assert col.spitzer_resistivity(1000.0, ln_lambda=17.0) == pytest.approx(17 * eta)
    ratio = col.spitzer_resistivity(1000.0, z_eff=2.0, ln_lambda=1.0) / eta
    assert ratio == pytest.approx(2 * 0.4408 / 0.5129)
    # Z -> infinity: alpha_0 -> 0.2949
    assert col.spitzer_resistivity(1000.0, z_eff=1e6, ln_lambda=1.0) / (
        1e6 * eta
    ) == pytest.approx(0.2949 / 0.5129, rel=1e-4)


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
    assert col.spitzer_resistivity(t, ln_lambda=lnl) == pytest.approx(
        expected, rel=1e-12
    )


def test_slowing_down_time_spitzer_value():
    n_cm3, t_ev, a, lnl = 1e13, 1000.0, 2.0, 17.0
    tau = col.slowing_down_time(
        n_cm3 * PER_CM3, t_ev, beam_mass_number=a, ln_lambda=lnl
    )
    assert tau == pytest.approx(6.3e8 * a * t_ev**1.5 / (n_cm3 * lnl), rel=3e-3)
    # scalings: ~ A_b / Z_b^2 / n_e, T_e^3/2
    assert col.slowing_down_time(2e19, t_ev, a, ln_lambda=lnl) == pytest.approx(tau / 2)
    assert col.slowing_down_time(1e19, t_ev, a, 2.0, ln_lambda=lnl) == pytest.approx(
        tau / 4
    )
    assert col.slowing_down_time(1e19, t_ev, 2 * a, ln_lambda=lnl) == pytest.approx(
        2 * tau
    )


def test_critical_energy_values_and_scalings():
    # Wesson: E_c = 14.8 T_e A_b [sum n_j Z_j^2/(n_e A_j)]^(2/3)
    assert col.critical_energy(1.0) == pytest.approx(14.8, rel=1e-3)
    assert col.critical_energy(1000.0, 2.0, 2.0) == pytest.approx(
        14.8 * 1000 * 2 * 0.5 ** (2 / 3), rel=1e-3
    )
    # hydrogenic species list equals the single-species form
    ions = [(0.5, 1, 2), (0.5, 1, 2)]
    assert col.critical_energy(1e3, 2.0, ions=ions) == pytest.approx(
        col.critical_energy(1e3, 2.0, 2.0)
    )
    # impurity: carbon 6+ with n_C/n_e = 0.1 (Z_eff-ish) raises the sum: 0.4 D + 0.1 C(6, A=12) -> check by hand
    s = 0.4 / 2 + 0.1 * 36 / 12
    assert col.critical_energy(
        1e3, 2.0, ions=[(0.4, 1, 2), (0.1, 6, 12)]
    ) == pytest.approx(14.8 * 1e3 * 2.0 * s ** (2 / 3), rel=1e-3)
    # beam ion with equal speed to the critical one: E_c = 1/2 m_b v_c^2, v_c^3 from the definition
    t, a = 800.0, 3.0
    v_te = math.sqrt(2 * t * par.ev_to_joule / par.electron_mass)
    v_c = (
        3 * math.sqrt(math.pi) / 4 * par.electron_mass / par.proton_mass * v_te**3
    ) ** (1 / 3)  # hydrogen target
    assert col.critical_energy(t, a) * par.ev_to_joule == pytest.approx(
        0.5 * a * par.proton_mass * v_c**2, rel=1e-12
    )


def test_slowing_down_energy_time_roundtrip_and_limits():
    n, t, a, lnl = 1e19, 2000.0, 2.0, 16.0
    e_c = col.critical_energy(t, a, 2.0)
    tau = col.slowing_down_time(n, t, a, ln_lambda=lnl)
    kw = dict(
        density=n, temperature=t, beam_mass_number=a, ion_mass_number=2.0, ln_lambda=lnl
    )
    e0 = 100e3
    for e in (80e3, 30e3, e_c, 5e3):
        time = col.slowing_down_time(
            n, t, a, ln_lambda=lnl, energy=e0, final_energy=e, critical_energy_ev=e_c
        )
        assert col.beam_stopping_fast_ion(e0, time=time, **kw) == pytest.approx(
            e, rel=1e-9
        )
    # thermalisation time: finite, and the energy is 0 afterwards
    t_th = col.slowing_down_time(
        n, t, a, ln_lambda=lnl, energy=e0, critical_energy_ev=e_c
    )
    assert col.beam_stopping_fast_ion(e0, time=1.01 * t_th, **kw) == 0.0
    # electron drag only (E_c -> 0): v = v0 exp(-t/tau_s), E = E0 exp(-2t/tau_s)
    kw0 = dict(kw, ion_mass_number=1e12)
    assert col.beam_stopping_fast_ion(e0, time=0.3 * tau, **kw0) == pytest.approx(
        e0 * math.exp(-0.6), rel=1e-6
    )
    t_e = col.slowing_down_time(
        n, t, a, ln_lambda=lnl, energy=e0, final_energy=e0 / 10, critical_energy_ev=1e-9
    )
    assert t_e == pytest.approx(tau / 2 * math.log(10), rel=1e-6)
    # no ion drag with an infinite critical energy is not what np.inf means: time diverges to tau/3 * ln(1)
    assert (
        col.slowing_down_time(
            n, t, a, ln_lambda=lnl, energy=e0, critical_energy_ev=np.inf
        )
        == 0.0
    )


def test_beam_stopping_rates():
    n, t, a, lnl = 1e19, 1000.0, 2.0, 17.0
    kw = dict(
        density=n, temperature=t, beam_mass_number=a, ion_mass_number=2.0, ln_lambda=lnl
    )
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
    assert col.beam_stopping_fast_ion(np.array([20e3, 80e3]), **kw)["total"].shape == (
        2,
    )


def _rk4_slowing_down(
    e0_ev, density, temperature, a_b, z_b, ions, ln_lambda, e_end_ev=None, xi0=1.0
):
    """Independent RK4 of the drag + Lorentz-diffusion moment ODEs (arXiv:2111.11721, Eq. D2-D7).

    Uses ln v as the independent variable, with nu_d = Z_eff Gamma / v^3, the electron drag nu_s of
    Eq. D6 and v_c^3 = 3 sqrt(pi)/4 (m_e/m_p) v_Te^3 sum_j f_j Z_j^2 / A_j, and integrates
    d<xi>/dt = -nu_d <xi>, d<xi^2>/dt = nu_d (1 - 3 <xi^2>), d(path)/dt = v <xi>, dt = dv / (dv/dt).
    Returns (<xi>, <xi^2>, path, time) at e_end_ev (default: ~rest).
    """
    me, mp, e = par.electron_mass, par.proton_mass, par.elementary_charge
    m_b = a_b * mp
    gamma = (
        density
        * z_b**2
        * e**4
        * ln_lambda
        / (4 * math.pi * par.vacuum_permittivity**2 * m_b**2)
    )
    z_eff = sum(f * z**2 for f, z, _ in ions)
    v_te = math.sqrt(2 * temperature * par.ev_to_joule / me)
    nu_s0 = 4 * m_b / (3 * math.sqrt(math.pi) * me) * gamma / v_te**3  # Eq. D6
    s_ion = sum(f * z**2 / a for f, z, a in ions)
    vc3 = 3 * math.sqrt(math.pi) / 4 * (me / mp) * v_te**3 * s_ion
    v0 = math.sqrt(2 * e0_ev * par.ev_to_joule / m_b)
    v_end = v0 * math.sqrt(e_end_ev / e0_ev) if e_end_ev else (vc3 ** (1 / 3)) * 1e-4

    def rhs(lnv, y):
        v = math.exp(lnv)
        nu_d = z_eff * gamma / v**3
        dvdt = -nu_s0 * v * (1 + vc3 / v**3)
        m1, m2, path, t = y
        # dy/dlnv = v dy/dt / (dv/dt)
        return np.array([-nu_d * m1, nu_d * (1 - 3 * m2), v * m1, 1.0]) * (v / dvdt)

    n = 6000
    x, y = math.log(v0), np.array([xi0, xi0**2, 0.0, 0.0])
    h = (math.log(v_end) - x) / n
    for _ in range(n):
        k1 = rhs(x, y)
        k2 = rhs(x + h / 2, y + h / 2 * k1)
        k3 = rhs(x + h / 2, y + h / 2 * k2)
        k4 = rhs(x + h, y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h
    return y[0], y[1], y[2], y[3]


PITCH_CASES = {
    "D in D": (2.0, [(1.0, 1, 2.0)]),
    "H in D": (1.0, [(1.0, 1, 2.0)]),
    "D in D + 5% C6+": (2.0, [(0.7, 1, 2.0), (0.05, 6, 12.0)]),
}


@pytest.mark.parametrize("name", list(PITCH_CASES))
def test_pitch_evolution_and_parallel_path_match_rk4(name):
    a_b, ions = PITCH_CASES[name]
    n, t, lnl, e0, xi0 = 1e19, 1500.0, 17.0, 80e3, 0.7
    e_c = col.critical_energy(t, a_b, ions=ions)
    beta = col.pitch_scattering_exponent(a_b, ions=ions)
    z_eff = sum(f * z**2 for f, z, _ in ions)
    s = sum(f * z**2 / a for f, z, a in ions)
    assert beta == pytest.approx(z_eff / (a_b * s), rel=1e-14)
    # the electron drag rate of D6 is 1 / tau_s of this module, so the RK4 above is on the same scale
    for e_end in (50e3, e_c, 2e3):
        m1, m2, _, time = _rk4_slowing_down(e0, n, t, a_b, 1, ions, lnl, e_end, xi0)
        mom = col.pitch_angle_moments(e_end, e0, e_c, beta, xi0)
        assert mom["mean"] == pytest.approx(m1, rel=2e-7)
        assert mom["mean_square"] == pytest.approx(m2, rel=2e-7)
        assert mom["width"] == pytest.approx(math.sqrt(m2 - m1**2), rel=2e-6)
        tau = col.slowing_down_time(
            n,
            t,
            a_b,
            ln_lambda=lnl,
            energy=e0,
            final_energy=e_end,
            critical_energy_ev=e_c,
        )
        assert tau == pytest.approx(time, rel=2e-7)
    # to rest: the path along B
    _, _, path, _ = _rk4_slowing_down(e0, n, t, a_b, 1, ions, lnl, None, xi0)
    kw = dict(
        density=n,
        temperature=t,
        beam_mass_number=a_b,
        ions=ions,
        ln_lambda=lnl,
        initial_pitch=xi0,
    )
    assert col.fast_ion_parallel_path(e0, **kw) == pytest.approx(path, rel=1e-5)


def test_pitch_evolution_limits():
    e_c = 20e3
    # above E_c the pitch is kept, far below it becomes isotropic: <xi^2> -> 1/3, <xi> -> 0
    early = col.pitch_angle_moments(0.99 * 1e6, 1e6, e_c, 1.0)
    assert early["mean"] == pytest.approx(1.0, abs=1e-3) and early["width"] < 0.02
    late = col.pitch_angle_moments(1e-3, 1e5, e_c, 1.0)
    assert late["mean"] < 1e-3 and late["mean_square"] == pytest.approx(1 / 3, abs=1e-3)
    # exponent beta = 0 (no scattering): nothing changes
    assert col.pitch_angle_moments(1e3, 8e4, e_c, 0.0, 0.5)["mean"] == pytest.approx(
        0.5
    )
    # more Z_eff, faster scattering: smaller mean pitch
    assert (
        col.pitch_angle_moments(1e4, 8e4, e_c, 2.5)["mean"]
        < col.pitch_angle_moments(1e4, 8e4, e_c, 1.0)["mean"]
    )
    # beta of a beam in its own isotope is 1 for any charge
    assert col.pitch_scattering_exponent(4.0, 4.0, 2.0) == pytest.approx(1.0)
    assert col.pitch_scattering_exponent(1.0, 2.0) == pytest.approx(2.0)


def test_shielding_and_beam_driven_current():
    assert (
        col.electron_shielding_factor(1.0, 1.0) == 0.0
    )  # complete cancellation, Z_b = Z_eff
    assert col.electron_shielding_factor(1.0, 2.0) == pytest.approx(0.5)
    assert col.electron_shielding_factor(1.0, 2.0, g=1.0) == pytest.approx(1.0)
    # L31 with no trapped electrons is 0; banana-regime value of the polynomial at f_t = 1, Z = 1
    assert col.bootstrap_coefficient_l31(1.5, 0.0) == 0.0
    assert col.bootstrap_coefficient_l31(1.0, 1.0) == pytest.approx(
        1 + 1.4 / 2 - 1.9 / 2 + 0.3 / 2 + 0.2 / 2
    )
    # collisionality lowers it
    assert col.bootstrap_coefficient_l31(1.0, 0.3, 1.0) < col.bootstrap_coefficient_l31(
        1.0, 0.3
    )
    kw = dict(
        density=1e19,
        temperature=1000.0,
        beam_mass_number=2.0,
        ion_mass_number=2.0,
        ln_lambda=17.0,
    )
    path = col.fast_ion_parallel_path(80e3, **kw)
    out = col.beam_driven_current(1e20, 1.7, 80e3, ion_charge_number=1.0, **kw)
    assert out["fast_ion"] == pytest.approx(
        par.elementary_charge * 1e20 * path / (2 * math.pi * 1.7)
    )
    assert out["net"] == 0.0  # Z_b = Z_eff = 1, G = 0
    out = col.beam_driven_current(1e20, 1.7, 80e3, g=0.3, **kw)
    assert out["net"] == pytest.approx(0.3 * out["fast_ion"])
