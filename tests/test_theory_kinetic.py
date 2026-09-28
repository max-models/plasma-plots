"""Kinetic dispersion relations of plasma_plots.theory.kinetic: literature values, limits and
independent checks with scipy (the Faddeeva function wofz and polynomial roots)."""

import warnings

import numpy as np
import pytest

from plasma_plots.theory import kinetic
from plasma_plots.theory.kinetic import (
    Maxwellian,
    beam_plasma_cold,
    bohm_gross,
    bump_on_tail,
    electrostatic_dielectric,
    ion_acoustic,
    ion_acoustic_fluid,
    landau_damping_weak,
    langmuir,
    maximum_growth,
    solve_dispersion,
    susceptibility,
    two_stream,
    two_stream_cold,
    weibel,
)


def scipy_dielectric(omega, k, species):
    """ε(ω, k) from scipy's Faddeeva function, independent of plasma_plots.theory.special."""
    special = pytest.importorskip("scipy.special")
    total = 1.0 + 0j
    with np.errstate(all="ignore"):
        for s in species:
            zeta = (omega - k * s.drift) / (np.sqrt(2) * abs(k) * s.thermal_speed)
            z = 1j * np.sqrt(np.pi) * special.wofz(zeta)
            total = total + s.density * s.charge**2 / s.mass / (k * s.thermal_speed) ** 2 * (1 + zeta * z)
    return total


def scipy_root(function, guess):
    optimize = pytest.importorskip("scipy.optimize")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return complex(optimize.newton(function, guess, tol=1e-13, maxiter=200))


# ---------------------------------------------------------------------------------------------
# The dielectric function
# ---------------------------------------------------------------------------------------------
def test_dielectric_agrees_with_scipy_near_and_far():
    """Including the asymptotic series (|ζ| ≥ 6) and the Stokes-smoothed Landau term."""
    rng = np.random.default_rng(1)
    omega = rng.uniform(-8, 8, 200) + 1j * rng.uniform(-1.5, 1.5, 200)
    k = rng.uniform(0.2, 2.0, 200)
    species = (Maxwellian(), Maxwellian(density=0.2, thermal_speed=0.4, drift=2.0))
    expected = scipy_dielectric(omega, k, species)
    np.testing.assert_allclose(electrostatic_dielectric(omega, k, species), expected, rtol=1e-9)
    # large |ζ| on and near the real axis: scipy is accurate here to ~1e-13 · |ζ|² relative
    zeta = np.linspace(6.5, 40, 50) + 1j * np.linspace(-0.5, 0.5, 50)[:, None]
    species = Maxwellian()
    k = 1.0
    np.testing.assert_allclose(susceptibility(zeta * np.sqrt(2), k, species),
                               scipy_dielectric(zeta * np.sqrt(2), k, [species]) - 1, rtol=1e-9)


def test_dielectric_derivative_and_limits():
    species = (Maxwellian(), Maxwellian(density=0.1, thermal_speed=0.5, drift=4.5))
    omega = np.array([0.3 + 0.1j, 1.2 - 0.4j, 2.0 + 0.0j, 5.0 - 0.2j, 20.0 + 1j])
    for k in (0.3, -0.7):
        h = 1e-6
        up, down = electrostatic_dielectric(omega + h, k, species), electrostatic_dielectric(omega - h, k, species)
        analytic = electrostatic_dielectric(omega, k, species, derivative=1)
        np.testing.assert_allclose(analytic, (up - down) / (2 * h), rtol=1e-7, atol=1e-9)
    # cold limit χ → −ω_p²/(ω − ku)², also for a drifting species
    assert susceptibility(3.0, 0.01, Maxwellian(drift=100.0, thermal_speed=1e-3)) == pytest.approx(-1 / 4, rel=1e-8)
    # χ(ω, −k) is χ(ω, k) with the drift reversed
    beam = Maxwellian(density=0.3, drift=2.0)
    reversed_beam = Maxwellian(density=0.3, drift=-2.0)
    assert susceptibility(1.1 + 0.2j, -0.4, beam) == pytest.approx(susceptibility(1.1 + 0.2j, 0.4, reversed_beam))
    # static limit: Debye shielding ε(0, k) = 1 + 1/k²
    assert electrostatic_dielectric(0.0, 0.5) == pytest.approx(5.0)
    assert Maxwellian().plasma_frequency == 1.0
    ions = Maxwellian.ions(mass_ratio=100.0, temperature_ratio=4.0, charge=2.0)
    assert ions.density == 0.5 and ions.thermal_speed == pytest.approx(0.05)
    assert ions.plasma_frequency == pytest.approx(np.sqrt(0.5 * 4 / 100))
    with pytest.raises(ValueError, match="derivative"):
        susceptibility(1.0, 1.0, Maxwellian(), derivative=2)


# ---------------------------------------------------------------------------------------------
# Langmuir waves
# ---------------------------------------------------------------------------------------------
LITERATURE = {  # Canosa (1973) and textbooks, truncated to 4 decimals
    0.3: 1.1598 - 0.0126j,
    0.4: 1.2850 - 0.0661j,
    0.5: 1.4156 - 0.1533j,
    1.0: 2.0459 - 0.8513j,
}


def test_langmuir_literature_values():
    k = np.array(list(LITERATURE))
    omega = langmuir(k)
    expected = np.array(list(LITERATURE.values()))
    # the tabulated digits are truncated, so each part is within 1e-4 below/above
    assert np.all(np.abs(omega.real - expected.real) < 1e-4)
    assert np.all(np.abs(omega.imag - expected.imag) < 1e-4)
    assert langmuir(0.5) == pytest.approx(1.4156618886045 - 0.1533594669096j, abs=1e-12)


def test_langmuir_agrees_with_scipy_roots():
    for k in (0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        root = scipy_root(lambda w: scipy_dielectric(w, k, [Maxwellian()]), langmuir(k) + 1e-3)
        assert langmuir(k) == pytest.approx(root, rel=1e-11, abs=1e-14)
    k = np.linspace(0.01, 3.0, 200)
    omega = langmuir(k)
    assert np.all(np.isfinite(omega))
    assert np.abs(electrostatic_dielectric(omega, k)).max() < 1e-12
    assert np.all(omega.imag < 1e-300)  # Landau damping, also at small k where it is ~1e-80 (or underflows)
    assert np.all(np.diff(omega.imag[k > 0.05]) < 0) and np.abs(np.diff(omega)).max() < 0.05  # one smooth branch
    assert langmuir(-0.4) == langmuir(0.4)
    assert langmuir(0.0) == 1.0
    assert isinstance(langmuir(0.3), complex) and langmuir([[0.3]]).shape == (1, 1)


def test_bohm_gross_and_weak_damping_limits():
    # ω = 1 + 3k²/2 + 15k⁴/8 + ...: the Bohm–Gross frequency misses 3k⁴
    for k in (0.02, 0.05):
        assert (langmuir(k).real - bohm_gross(k).real) / k**4 == pytest.approx(3.0, rel=0.05)
    assert bohm_gross(0.5) == pytest.approx(np.sqrt(1.75))
    # the weak-damping rate converges to the exact one as k → 0
    errors = [abs(landau_damping_weak(k).imag / langmuir(k).imag - 1) for k in (0.08, 0.1, 0.15, 0.2)]
    assert errors[1] < 0.05 and errors == sorted(errors)
    assert landau_damping_weak(0.3).imag == pytest.approx(-np.sqrt(np.pi / 8) / 0.027 * np.exp(-1 / 0.18 - 1.5))
    assert landau_damping_weak(0.0) == 1.0


# ---------------------------------------------------------------------------------------------
# Ion-acoustic waves
# ---------------------------------------------------------------------------------------------
def test_ion_acoustic_sound_speed_and_damping():
    mass_ratio = kinetic.PROTON_ELECTRON_MASS_RATIO
    # T_e ≫ T_i: ω/k → c_s = √(T_e/m_i) with electron Landau damping γ/ω = −√(π m_e/(8 m_i))
    omega = ion_acoustic(0.01, temperature_ratio=1000.0, mass_ratio=mass_ratio)
    assert omega.real / 0.01 * np.sqrt(mass_ratio) == pytest.approx(1.0, rel=5e-3)
    assert omega.imag / omega.real == pytest.approx(-np.sqrt(np.pi / (8 * mass_ratio)), rel=0.03)
    # the fluid estimate with γ_i = 3 at T_e/T_i = 100, including the Debye correction at larger k
    k = np.array([0.01, 0.1, 0.5])
    np.testing.assert_allclose(ion_acoustic(k, 100.0).real, ion_acoustic_fluid(k, 100.0).real, rtol=0.01)
    # stronger ion Landau damping for T_e ~ T_i
    hot_ions, cold_ions = ion_acoustic(0.1, 1.0), ion_acoustic(0.1, 10.0)
    assert hot_ions.imag / hot_ions.real < 5 * cold_ions.imag / cold_ions.real


def test_ion_acoustic_roots_are_zeros():
    k, tau = np.linspace(0.02, 2.0, 25)[:, None], np.array([0.5, 1.0, 3.0, 30.0])
    omega = ion_acoustic(k, tau, 100.0)
    assert omega.shape == (25, 4) and np.all(np.isfinite(omega))
    for i, t in enumerate(tau):
        species = [Maxwellian(), Maxwellian.ions(100.0, t)]
        residual = scipy_dielectric(omega[:, i], k[:, 0], species)
        assert np.abs(residual).max() < 1e-8
    assert np.all(omega.imag < 0) and np.all(omega.real > 0)


# ---------------------------------------------------------------------------------------------
# Beams and streams
# ---------------------------------------------------------------------------------------------
def test_two_stream_cold_matches_polynomial_roots():
    for k, v, n in [(0.3, 1.0, 0.5), (0.9, 1.0, 0.5), (1.5, 1.0, 0.5), (0.2, 3.0, 0.1)]:
        a = k * v
        # (ω − a)²(ω + a)² − n (ω + a)² − n (ω − a)² = 0
        minus, plus = np.poly1d([1, -a]) ** 2, np.poly1d([1, a]) ** 2
        polynomial = minus * plus - n * plus - n * minus
        expected = np.sort_complex(np.roots(polynomial.coeffs))
        got = np.sort_complex(two_stream_cold(k, v, n, all_roots=True))
        np.testing.assert_allclose(got, expected, atol=1e-9)
        slow = two_stream_cold(k, v, n)
        assert slow.imag == pytest.approx(max(np.roots(polynomial.coeffs).imag), abs=1e-9)
    assert two_stream_cold(0.3, 1.0).real == 0 and two_stream_cold(1.5, 1.0).imag == 0
    k_max, omega = maximum_growth(lambda k: two_stream_cold(k, 1.0), (0.01, 1.4))
    assert k_max == pytest.approx(np.sqrt(3 / 8), abs=1e-6) and omega.imag == pytest.approx(np.sqrt(0.5) / 2, rel=1e-10)


def test_beam_plasma_cold_matches_polynomial_roots():
    for k, v, nb in [(1.0, 1.0, 0.001), (0.5, 2.0, 0.1), (3.0, 1.0, 0.01)]:
        a = k * v
        # ω²(ω − a)² − (ω − a)² − n_b ω² = 0
        square, shifted = np.poly1d([1, 0, 0]), np.poly1d([1, -a]) ** 2
        polynomial = square * shifted - shifted - nb * square
        expected = np.roots(polynomial.coeffs)
        got = beam_plasma_cold(k, v, nb, all_roots=True)
        np.testing.assert_allclose(np.sort_complex(got), np.sort_complex(expected), atol=1e-9)
        omega = beam_plasma_cold(k, v, nb)
        if expected.imag.max() > 1e-8:
            assert omega == pytest.approx(expected[np.argmax(expected.imag)], abs=1e-9)
        else:
            assert omega.imag == 0
    # weak beam: γ_max ≈ (√3/2)(n_b/2)^(1/3) at k v_b ≈ ω_pe
    nb = 1e-5
    _, omega = maximum_growth(lambda k: beam_plasma_cold(k, 1.0, nb), (0.9, 1.1))
    assert omega.imag == pytest.approx(np.sqrt(3) / 2 * (nb / 2) ** (1 / 3), rel=0.02)


def test_two_stream_kinetic():
    k = np.linspace(0.05, 0.3, 11)
    # cold limit (near the cold cutoff k v_b = 1 the warm beams grow faster than the cold ones)
    np.testing.assert_allclose(two_stream(k, 3.0, 1e-4), two_stream_cold(k, 3.0), rtol=1e-6)
    # growth decreases with temperature and the roots are zeros of ε
    warm = two_stream(k, 3.0, 0.3)
    assert np.all(warm.imag[:8] < two_stream_cold(k[:8], 3.0).imag) and np.all(warm.real == 0)
    species = [Maxwellian(density=0.5, thermal_speed=0.3, drift=d) for d in (3.0, -3.0)]
    assert np.abs(scipy_dielectric(warm, k, species)).max() < 1e-10
    # no instability once the beams overlap (v_b ≲ 1.3 v_t): the least damped root is a Langmuir wave
    stable = two_stream(k, 1.0, 1.0)
    assert np.all(stable.imag < 1e-20) and np.all(stable.real > 1)
    species = [Maxwellian(density=0.5, thermal_speed=1.0, drift=d) for d in (1.0, -1.0)]
    assert np.abs(scipy_dielectric(stable, k, species)).max() < 1e-10
    # γ(k) continuous through the marginal wavenumber
    kk = np.linspace(0.2, 0.5, 61)
    gamma = two_stream(kk, 3.0, 0.5).imag
    # (γ ∝ √(k_c − k) just below the marginal k_c, where iγ meets its mirror −iγ)
    assert gamma[0] > 0 and gamma[-1] < 0 and np.abs(np.diff(gamma)).max() < 0.1
    assert np.sum(np.diff(np.sign(gamma)) != 0) == 1


def test_bump_on_tail():
    k = np.linspace(0.15, 0.4, 11)
    omega = bump_on_tail(k)
    assert np.all(omega.imag > 0)
    species = [Maxwellian(density=0.9), Maxwellian(density=0.1, thermal_speed=0.5, drift=4.5)]
    assert np.abs(scipy_dielectric(omega, k, species)).max() < 1e-10
    # it is the most unstable root: no root from a dense grid of guesses grows faster
    for kk, w in zip(k[::5], omega[::5]):
        for guess in (np.linspace(0.2, 2.5, 24)[:, None] + 1j * np.array([-0.2, 0.05, 0.3])).ravel():
            try:
                root = scipy_root(lambda x: scipy_dielectric(x, kk, species), complex(guess))
            except RuntimeError:
                continue
            if abs(scipy_dielectric(root, kk, species)) < 1e-9:
                assert root.imag <= w.imag + 1e-9
    # without the beam: the (damped) Langmuir wave
    assert bump_on_tail(0.3, beam_density=0.0) == pytest.approx(langmuir(0.3), abs=1e-10)
    assert np.all(bump_on_tail(k, beam_density=0.0).imag < 0)
    # mirrored for negative k, broadcasting over parameters
    assert bump_on_tail(-0.3) == pytest.approx(-bump_on_tail(0.3).conjugate())
    assert bump_on_tail(k[:, None], beam_density=[0.0, 0.05, 0.1]).shape == (11, 3)
    k_max, omega_max = maximum_growth(bump_on_tail, (0.1, 0.6))
    assert 0.2 < k_max < 0.35 and omega_max.imag >= omega.imag.max()


# ---------------------------------------------------------------------------------------------
# Weibel
# ---------------------------------------------------------------------------------------------
def test_weibel_cutoff_and_dispersion():
    special = pytest.importorskip("scipy.special")
    for anisotropy, speed in [(4.0, 0.1), (2.0, 0.3), (10.0, 0.05)]:
        cutoff = np.sqrt(anisotropy - 1)
        k = cutoff * np.array([0.1, 0.5, 0.9, 0.999, 1.001, 1.2, 2.0])
        omega = weibel(k, anisotropy, speed)
        assert np.all(omega.real == 0)
        np.testing.assert_array_equal(omega.imag > 0, k < cutoff)
        assert abs(weibel(cutoff, anisotropy, speed)) < 1e-12
        # the dispersion relation, with scipy's Faddeeva function
        zeta = omega / (np.sqrt(2) * k * speed)
        residual = omega**2 - k**2 - 1 + anisotropy * (1 + zeta * 1j * np.sqrt(np.pi) * special.wofz(zeta))
        assert np.abs(residual).max() < 1e-10
        # near the cutoff γ ≈ √(2/π) k v_∥ (A − 1 − k²)/A
        k = cutoff * 0.999
        approx = np.sqrt(2 / np.pi) * k * speed * (anisotropy - 1 - k**2) / anisotropy
        assert weibel(k, anisotropy, speed).imag == pytest.approx(approx, rel=0.01)
    # isotropic and T⊥ < T∥: stable
    assert np.all(weibel([0.5, 2.0], 1.0, 0.1).imag < 0) and np.all(weibel([0.5, 2.0], 0.5, 0.1).imag < 0)
    assert weibel(0.0, 4.0, 0.1) == 0 and weibel(-1.0, 4.0, 0.1) == weibel(1.0, 4.0, 0.1)
    assert weibel(np.linspace(0, 3, 5)[:, None], [2.0, 4.0, 9.0], 0.2).shape == (5, 3)


# ---------------------------------------------------------------------------------------------
# The root finder
# ---------------------------------------------------------------------------------------------
def test_solve_dispersion():
    k = np.linspace(0.2, 0.8, 13)
    continued = solve_dispersion(electrostatic_dielectric, k, 1.06)
    np.testing.assert_allclose(continued, langmuir(k), rtol=1e-11)
    analytic = solve_dispersion(electrostatic_dielectric, k, 1.06,
                                derivative=lambda w, kk: electrostatic_dielectric(w, kk, derivative=1))
    np.testing.assert_allclose(analytic, langmuir(k), rtol=1e-11)
    independent = solve_dispersion(electrostatic_dielectric, k, langmuir(k) + 0.01, continuation=False)
    np.testing.assert_allclose(independent, langmuir(k), rtol=1e-11)
    assert isinstance(solve_dispersion(electrostatic_dielectric, 0.5, 1.4), complex)
    # no root: nan, and no exception
    assert np.all(np.isnan(solve_dispersion(lambda w, kk: 1 + 0 * w, k, 1.0)))
    assert np.isnan(solve_dispersion(lambda w, kk: w**2 + 1, 1.0, 0.0))  # zero derivative at the guess
    with pytest.raises(ValueError, match="one-dimensional"):
        solve_dispersion(electrostatic_dielectric, k[:, None], 1.0)
    with pytest.raises(ValueError, match="scalar"):
        solve_dispersion(electrostatic_dielectric, k, np.ones(13))
    with pytest.raises(ValueError, match="no finite"):
        maximum_growth(lambda k: np.full(np.shape(k), complex(np.nan, np.nan)), (0.1, 1.0))
