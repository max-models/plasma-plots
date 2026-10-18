"""plasma_plots.theory.magnetron: Hull cutoff against Boris orbits, Brillouin and
critical-velocity identities, and the gradient-drift roots against their limits."""

import numpy as np
import pytest

from plasma_plots.theory import magnetron, parameters
from plasma_plots.theory.parameters import electron_mass, elementary_charge, proton_mass


def coaxial_reaches_anode(field, ra, rc, voltage, steps=200000):
    """Electron at rest at r_c in the log field of coaxial cylinders, axial B (Boris, 2D)."""
    q, m = -elementary_charge, electron_mass
    omega = elementary_charge * field / m
    dt = 0.02 / omega
    x, v = np.array([rc, 0.0, 0.0]), np.zeros(3)
    t_vec = q * np.array([0, 0, field]) * dt / (2 * m)
    s_vec = 2 * t_vec / (1 + t_vec @ t_vec)
    scale = voltage / np.log(ra / rc)
    for _ in range(steps):
        r2 = x[0] ** 2 + x[1] ** 2
        e = -scale * np.array([x[0], x[1], 0.0]) / r2  # E = -grad(phi), phi = V ln(r/rc)/ln(ra/rc)
        v_minus = v + q * e * dt / (2 * m)
        v_prime = v_minus + np.cross(v_minus, t_vec)
        v = v_minus + np.cross(v_prime, s_vec) + q * e * dt / (2 * m)
        x = x + v * dt
        r = np.hypot(x[0], x[1])
        if r >= ra:
            return True
    return False


def test_hull_planar_formula():
    d, v = 0.05, 300.0
    assert magnetron.hull_cutoff(voltage=v, gap=d) == pytest.approx(
        np.sqrt(2 * electron_mass * v / elementary_charge) / d
    )
    assert magnetron.hull_cutoff(field=magnetron.hull_cutoff(voltage=v, gap=d), gap=d) == pytest.approx(v)


def test_hull_cylindrical_reduces_to_planar():
    d = 1e-3
    for ra in (1.0, 0.5):
        rc = ra - d
        assert magnetron.hull_cutoff(voltage=100.0, anode_radius=ra, cathode_radius=rc) == pytest.approx(
            magnetron.hull_cutoff(voltage=100.0, gap=d), rel=2 * d / ra
        )


def test_hull_cylindrical_formula():
    ra, rc, v = 0.06, 0.01, 300.0
    expected = np.sqrt(8 * electron_mass * v / elementary_charge) / (ra * (1 - (rc / ra) ** 2))
    assert magnetron.hull_cutoff(voltage=v, anode_radius=ra, cathode_radius=rc) == pytest.approx(expected)


@pytest.mark.parametrize("ra, rc", [(0.06, 0.01), (0.03, 0.02)])
def test_hull_cylindrical_orbit(ra, rc):
    """An electron from rest reaches the anode below B_c and is confined above it."""
    v = 300.0
    bc = magnetron.hull_cutoff(voltage=v, anode_radius=ra, cathode_radius=rc)
    assert coaxial_reaches_anode(0.97 * bc, ra, rc, v)
    assert not coaxial_reaches_anode(1.03 * bc, ra, rc, v)


def test_hull_planar_orbit():
    """The coaxial test in the thin-gap limit is the planar one; check with a very thin gap."""
    rc, d, v = 1.0, 0.005, 100.0
    bc = magnetron.hull_cutoff(voltage=v, gap=d)
    assert coaxial_reaches_anode(0.95 * bc, rc + d, rc, v, steps=400000)
    assert not coaxial_reaches_anode(1.05 * bc, rc + d, rc, v, steps=400000)


def test_hull_errors():
    with pytest.raises(ValueError):
        magnetron.hull_cutoff(gap=0.05)
    with pytest.raises(ValueError):
        magnetron.hull_cutoff(voltage=1.0, field=1.0, gap=0.05)
    with pytest.raises(ValueError):
        magnetron.hull_cutoff(voltage=1.0)


def test_brillouin_flow():
    flow = magnetron.brillouin_flow(0.05, radius=0.02)
    assert parameters.plasma_frequency(flow["density"]) ** 2 == pytest.approx(
        0.5 * parameters.cyclotron_frequency(0.05) ** 2
    )
    assert flow["angular_frequency"] == pytest.approx(parameters.cyclotron_frequency(0.05) / 2)
    assert flow["speed"] == pytest.approx(0.02 * flow["angular_frequency"])


def test_hall_parameter_and_ionisation_velocity():
    assert magnetron.hall_parameter(0.05, 1e8) == pytest.approx(parameters.cyclotron_frequency(0.05) / 1e8)
    assert magnetron.hall_parameter(0.05, 1e8, proton_mass * 40) < 2e-3
    assert magnetron.critical_ionisation_velocity(100.0, 39.948) == pytest.approx(21.9e3, rel=2e-3)
    half = 0.5 * 39.948 * proton_mass * magnetron.critical_ionisation_velocity(100.0, 39.948) ** 2
    assert half == pytest.approx(100.0 * elementary_charge)


def test_exb_drift_speed():
    assert magnetron.exb_drift_speed(1e4, 0.05) == pytest.approx(2e5)
    assert magnetron.exb_drift_speed(np.array([1e4, -2e4]), 0.05) == pytest.approx([2e5, 4e5])


K, E0, B, LN, A = 2 * np.pi / 0.05, 1e4, 0.05, 0.01, 39.948


def test_simon_hoh_solves_quadratic_and_phase_velocity():
    w = magnetron.simon_hoh_frequency(K, E0, B, LN, A)
    omega_i = parameters.cyclotron_frequency(B, A * proton_mass)
    a, omega_e = omega_i * K * LN, K * E0 / B
    assert w**2 == pytest.approx(a * (w - omega_e))
    assert w.imag > 0
    assert w.real / K == pytest.approx(omega_i * LN / 2)
    assert magnetron.simon_hoh_growth(K, E0, B, LN, A) == pytest.approx(np.sqrt(a * omega_e - a**2 / 4))
    # large-drift limit
    assert w.imag == pytest.approx(K * np.sqrt(omega_i * LN * E0 / B), rel=1e-2)


def test_simon_hoh_stable_when_field_opposes_gradient():
    assert magnetron.simon_hoh_growth(K, E0, B, -LN, A) == 0.0
    assert magnetron.simon_hoh_growth(K, -E0, B, LN, A) == 0.0


def test_gradient_drift_cold_limit_matches_simon_hoh():
    # T_e drops out; the remaining finite-rho_e correction is of order k^2 L v_E / omega_ce (0.4 %)
    w0 = magnetron.simon_hoh_frequency(K, E0, B, LN, A)
    for te in (1e-6, 1e-3):
        w = magnetron.gradient_drift_frequency(K, E0, B, LN, te, A)
        assert w == pytest.approx(w0, rel=1e-2)


def test_gradient_drift_finite_rho_and_collisions_perturb_slightly():
    w0 = magnetron.gradient_drift_frequency(K, E0, B, LN, 3.0, A)
    w1 = magnetron.gradient_drift_frequency(K, E0, B, LN, 3.0, A, collision_frequency=1e7)
    assert w0.imag == pytest.approx(magnetron.simon_hoh_growth(K, E0, B, LN, A), rel=0.05)
    assert abs(w1.imag - w0.imag) < 0.05 * w0.imag
    assert magnetron.gradient_drift_growth(K, E0, B, LN, 3.0, A) == pytest.approx(w0.imag)
    assert magnetron.gradient_drift_phase_velocity(K, E0, B, LN, 3.0, A) == pytest.approx(w0.real / K)


def test_gradient_drift_broadcasts():
    k = np.array([K, 2 * K])
    growth = magnetron.gradient_drift_growth(k, E0, B, LN, 3.0, A)
    assert growth.shape == (2,) and growth[1] > growth[0]


# --- lower-hybrid drift (Malkov and Sotnikov, arXiv:1805.01933, Eq. (1)) ----------------------
LHD = dict(field=0.02, density=1e17, electron_temperature=10.0, mass_number=131.293)


def lhd_plasma(field, density, electron_temperature, mass_number):
    e = elementary_charge
    wpe2 = density * e**2 / (parameters.vacuum_permittivity * electron_mass)
    wpi2 = wpe2 * electron_mass / (mass_number * proton_mass)
    wce = e * field / electron_mass
    return wpe2, wpi2, wce, 2 * electron_temperature * e / electron_mass


def test_lower_hybrid_drift_root_satisfies_equation_with_scipy():
    special = pytest.importorskip("scipy.special")
    k, ti, vi, ve, vd, kz = 1326.0, 10.0, -1e4, 0.0, 3e3, 0.0
    w = magnetron.lower_hybrid_drift(k, 0.02, 1e17, 10.0, ti, vi, ve, 131.293, kz, vd)
    wpe2, wpi2, wce, vte2 = lhd_plasma(0.02, 1e17, 10.0, 131.293)
    vti = np.sqrt(2 * ti * elementary_charge / (131.293 * proton_mass))
    b = k**2 * vte2 / (2 * wce**2)
    xi = (w - k * vi) / (k * vti)
    plasma_z = 1j * np.sqrt(np.pi) * special.wofz(xi)
    gamma0 = special.ive(0, b)
    value = (
        1
        + 2 * wpi2 / (k * vti) ** 2 * (1 + xi * plasma_z)
        + wpe2 / wce**2 * (1 - gamma0) / b
        - kz**2 * wpe2 / (w - k * ve) ** 2
        + 2 * wpe2 / (k**2 * vte2) * k * vd / (w - k * ve)
    )
    assert w.imag > 1e6  # unstable: the ion-resonant LHDI of the docstring example
    assert abs(value) < 1e-8


def test_lower_hybrid_drift_galilean_shift_and_mirror():
    args = (1326.0, 0.02, 1e17, 10.0, 10.0)
    base = magnetron.lower_hybrid_drift(*args, -1e4, 0.0, 131.293, 0.0, 3e3)
    shifted = magnetron.lower_hybrid_drift(*args, -1e4 + 5e3, 5e3, 131.293, 0.0, 3e3)
    assert shifted.imag == pytest.approx(base.imag, rel=1e-6)
    assert shifted.real == pytest.approx(base.real + 1326.0 * 5e3, rel=1e-6)
    mirrored = magnetron.lower_hybrid_drift(*args, 1e4, 0.0, 131.293, 0.0, -3e3)
    assert mirrored == pytest.approx(-base.conjugate(), rel=1e-6)


def test_lower_hybrid_drift_needs_a_drive():
    for ti in (0.5, 10.0):
        w = magnetron.lower_hybrid_drift(1326.0, 0.02, 1e17, 10.0, ti, 0.0, 3e5, 131.293)
        assert w.imag <= 1e-6 * abs(w)


def test_lower_hybrid_drift_cold_ions_match_polynomial_roots_and_threshold():
    k, kz = 1000.0, 0.05
    wpe2, wpi2, wce, vte2 = lhd_plasma(**LHD)
    b = k**2 * vte2 / (2 * wce**2)
    flr = 1 + wpe2 / wce**2 * (1 - np.i0(b) * np.exp(-b)) / b
    c2 = kz**2 * wpe2

    def modified_two_stream(ve):
        # R (w − ui)² (w − ue)² − ωpi² (w − ue)² − c² (w − ui)² = 0 with ui = 0, ue = k ve
        ue = k * ve
        poly = np.poly1d([flr, 0, 0]) * np.poly1d([1, -ue]) ** 2
        poly = poly - wpi2 * np.poly1d([1, -ue]) ** 2 - c2 * np.poly1d([1, 0, 0])
        roots = poly.roots
        return roots[np.argmax(roots.imag)]

    critical = np.sqrt((wpi2 ** (1 / 3) + c2 ** (1 / 3)) ** 3 / flr) / k  # Eq. (7)
    for fraction in (0.3, 0.8, 0.97):
        expected = modified_two_stream(fraction * critical)
        got = magnetron.lower_hybrid_drift(k, 0.02, 1e17, 10.0, 1e-6, 0.0, fraction * critical, 131.293, kz)
        assert expected.imag > 0
        assert got == pytest.approx(expected, rel=1e-5)
    for fraction in (1.02, 1.5):
        got = magnetron.lower_hybrid_drift(k, 0.02, 1e17, 10.0, 1e-6, 0.0, fraction * critical, 131.293, kz)
        assert got.imag <= 1e-6 * abs(got)


def test_lower_hybrid_drift_cold_threshold_growth_formula():
    # near-threshold growth of the source (just above Eq. (7)): γ = Δ² K^{1/6} √(F_min − R)
    # / (√3 (1 + K^{1/3})^{5/2} ω_pi), Δ = k (V_e − V_i), K = (m_i/m_e)(k_z/k)²
    k, kz = 1000.0, 0.05
    wpe2, wpi2, wce, vte2 = lhd_plasma(**LHD)
    b = k**2 * vte2 / (2 * wce**2)
    flr = 1 + wpe2 / wce**2 * (1 - np.i0(b) * np.exp(-b)) / b
    c2 = kz**2 * wpe2
    critical = np.sqrt((wpi2 ** (1 / 3) + c2 ** (1 / 3)) ** 3 / flr) / k
    drift = 0.995 * critical
    delta = k * drift
    x = np.linspace(1e-3, 1 - 1e-3, 400001) * delta
    f_min = np.min(wpi2 / x**2 + c2 / (delta - x) ** 2)
    mass_ratio = 131.293 * proton_mass / electron_mass
    big_k = mass_ratio * kz**2
    formula = (
        delta**2
        * big_k ** (1 / 6)
        / (np.sqrt(3) * (1 + big_k ** (1 / 3)) ** 2.5 * np.sqrt(wpi2))
        * np.sqrt(f_min - flr)
    )
    got = magnetron.lower_hybrid_drift(k, 0.02, 1e17, 10.0, 1e-6, 0.0, drift, 131.293, kz)
    assert got.imag == pytest.approx(formula, rel=0.03)
