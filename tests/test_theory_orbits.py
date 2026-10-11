"""plasma_plots.theory.orbits: gyromotion and drifts (also against full Lorentz orbits), and the
trapped-particle results against numerical bounce motion in B = B₀/(1 + ε cos θ)."""

import numpy as np
import pytest

from plasma_plots.theory import orbits
from plasma_plots.theory.special import elliptic_k


# ---------------------------------------------------------------------------------------------
# Helpers: a Boris pusher and the 1-D bounce motion along a field line
# ---------------------------------------------------------------------------------------------
def boris(x, v, field, charge, mass, dt, steps):
    """Push one particle in a static magnetic field ``field(x)``; return the positions."""
    positions = np.empty((steps + 1, 3))
    positions[0] = x
    for i in range(steps):
        t = charge * field(x) * dt / (2 * mass)
        s = 2 * t / (1 + t @ t)
        v_prime = v + np.cross(v, t)
        v = v + np.cross(v_prime, s)
        x = x + v * dt
        positions[i + 1] = x
    return positions


def bounce_orbit(kappa2, epsilon, q, gyrofrequency, dt, t_end):
    """RK4 for θ' = v∥/q, v∥' = −(λ/2) (dB/dθ)/q, r' = v_d sin θ (R₀ = v = B₀ = 1).

    B = 1/(1 + ε cos θ), λ = 1 + ε − 2εκ², and the vertical drift v_d = (v∥² + v⊥²/2)/Ω of the
    1/R field. Returns t and the states (θ, v∥, r), vectorized over ``kappa2``.
    """
    kappa2 = np.asarray(kappa2, dtype=float)
    lam = 1 + epsilon - 2 * epsilon * kappa2

    def rhs(state):
        theta, vpar, _ = state
        b = 1 / (1 + epsilon * np.cos(theta))
        db = epsilon * np.sin(theta) * b**2
        drift = (vpar**2 + lam * b / 2) / gyrofrequency
        return np.array([vpar / q, -lam / 2 * db / q, drift * np.sin(theta)])

    state = np.array([np.zeros_like(kappa2), np.sqrt(1 - lam / (1 + epsilon)), np.zeros_like(kappa2)])
    steps = int(t_end / dt)
    history = np.empty((steps + 1,) + state.shape)
    history[0] = state
    for i in range(steps):
        k1 = rhs(state)
        k2 = rhs(state + dt / 2 * k1)
        k3 = rhs(state + dt / 2 * k2)
        k4 = rhs(state + dt * k3)
        state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        history[i + 1] = state
    return dt * np.arange(steps + 1), history


def downward_crossings(t, y):
    """Times where y goes from positive to negative, linearly interpolated."""
    i = np.nonzero((y[:-1] > 0) & (y[1:] <= 0))[0]
    return t[i] + (t[i + 1] - t[i]) * y[i] / (y[i] - y[i + 1])


# ---------------------------------------------------------------------------------------------
# Gyromotion and drifts
# ---------------------------------------------------------------------------------------------
def test_gyromotion():
    assert orbits.gyrofrequency(2.0, charge=3.0, mass=4.0) == pytest.approx(1.5)
    assert orbits.gyrofrequency(-2.0, charge=-1.0) == pytest.approx(-2.0)
    v, b = np.array([1.0, 2.0]), np.array([[1.0], [4.0]])
    np.testing.assert_allclose(
        orbits.gyroradius(v, b, charge=-2.0) * np.abs(orbits.gyrofrequency(b, -2.0)),
        v * np.ones((2, 2)),
    )
    assert orbits.magnetic_moment(3.0, 2.0, mass=2.0) == pytest.approx(4.5)
    assert np.ndim(orbits.gyroradius(1.0, 1.0)) == 0


def test_exb_drift():
    rng = np.random.default_rng(1)
    E, B = rng.normal(size=(20, 3)), rng.normal(size=(20, 3))
    v = orbits.exb_drift(E, B)
    np.testing.assert_allclose(np.sum(v * E, axis=-1), 0, atol=1e-12)
    np.testing.assert_allclose(np.sum(v * B, axis=-1), 0, atol=1e-12)
    b2 = np.sum(B**2, axis=-1)
    e_perp = np.sqrt(np.sum(E**2, axis=-1) - np.sum(E * B, axis=-1) ** 2 / b2)
    np.testing.assert_allclose(np.linalg.norm(v, axis=-1), e_perp / np.sqrt(b2))
    np.testing.assert_allclose(orbits.exb_drift([1.0, 0, 0], [0, 0, 1.0]), [0, -1.0, 0])  # E × B
    assert orbits.exb_drift(E[:, None], B[None, :5]).shape == (20, 5, 3)
    with pytest.raises(ValueError, match="length 3"):
        orbits.exb_drift([1.0, 0.0], [0.0, 1.0])


def test_grad_b_drift_directions():
    B, grad_B = np.array([0.0, 0.0, 2.0]), np.array([0.3, 0.0, 0.0])
    ion = orbits.grad_b_drift(1.5, B, grad_B, charge=1.0, mass=2.0)
    electron = orbits.grad_b_drift(1.5, B, grad_B, charge=-1.0, mass=2.0)
    np.testing.assert_allclose(electron, -ion)
    mu = orbits.magnetic_moment(1.5, 2.0, mass=2.0)
    np.testing.assert_allclose(ion, [0, mu * 0.3 / (1.0 * 2.0), 0])  # μ ∇B / (qB) along B × ∇B for ions
    # broadcasting of the scalars against the vectors
    assert orbits.grad_b_drift(np.ones(4), B, grad_B).shape == (4, 3)


@pytest.mark.parametrize("charge", [1.0, -1.0])
def test_grad_b_drift_against_lorentz_orbit(charge):
    """B = (1 + x/L) ẑ: the mean velocity of a full orbit is the grad-B drift."""
    length = 100.0
    field = lambda x: np.array([0.0, 0.0, 1 + x[0] / length])
    v = np.array([0.0, 1.0, 0.0])
    x0 = -np.cross(v, [0, 0, 1.0]) / charge  # guiding center at the origin (m = B = 1 there)
    period = 2 * np.pi
    steps_per_period = 400
    positions = boris(x0, v, field, charge, 1.0, period / steps_per_period, 40 * steps_per_period)
    measured = (positions[-1] - positions[0]) / (40 * period)
    expected = orbits.grad_b_drift(1.0, [0, 0, 1.0], [1 / length, 0, 0], charge=charge)
    np.testing.assert_allclose(measured, expected, atol=2e-2 * np.linalg.norm(expected))


# ---------------------------------------------------------------------------------------------
# Trapped particles
# ---------------------------------------------------------------------------------------------
def test_trapped_fraction():
    assert orbits.trapped_fraction(0.0) == 0.0
    assert orbits.trapped_fraction(1.0) == 1.0
    small = np.array([1e-6, 1e-5, 1e-4])
    np.testing.assert_allclose(orbits.trapped_fraction(small) / (1.46 * np.sqrt(small)), 1, atol=2e-3)
    eps = np.linspace(0.01, 0.95, 30)
    exact = orbits.trapped_fraction(eps)
    assert np.all(np.diff(exact) > 0)
    np.testing.assert_allclose(orbits.trapped_fraction(eps, "lin-liu"), exact, rtol=2e-2)
    assert orbits.trapped_fraction(0.04, "sqrt") == pytest.approx(0.292)
    assert orbits.trapped_fraction(eps.reshape(5, 6)).shape == (5, 6)
    with pytest.raises(ValueError, match="approximation"):
        orbits.trapped_fraction(0.1, "linear")


def test_trapped_fraction_against_scipy():
    integrate = pytest.importorskip("scipy.integrate")
    for eps in [0.01, 0.2, 0.7]:

        def average(f):
            return integrate.quad(lambda t: f(t) * (1 + eps * np.cos(t)), 0, 2 * np.pi, limit=200)[0] / (2 * np.pi)

        b = lambda t: 1 / (1 + eps * np.cos(t))
        inner = integrate.quad(
            lambda lam: lam / average(lambda t: np.sqrt(max(1 - lam * b(t), 0))),
            0,
            1 - eps,
            limit=200,
        )[0]
        reference = 1 - 0.75 * average(lambda t: b(t) ** 2) * inner
        assert orbits.trapped_fraction(eps) == pytest.approx(reference, rel=1e-8)


def test_trapping_boundary_and_parameters():
    eps = np.array([0.05, 0.2, 0.5])
    lam_c = orbits.trapping_boundary(eps)
    np.testing.assert_allclose(lam_c, 1 / (1 / (1 - eps)))  # B₀ / B_max
    np.testing.assert_allclose(orbits.trapping_parameter(lam_c, eps), 1)
    np.testing.assert_allclose(orbits.trapping_parameter(1 + eps, eps), 0, atol=1e-15)
    for theta in [0.0, 1.0, 2.5]:
        pitch = orbits.trapping_boundary(eps, "pitch", theta=theta)
        np.testing.assert_allclose(orbits.pitch_parameter(pitch, eps, theta), lam_c)
    np.testing.assert_allclose(orbits.trapping_boundary(eps, "pitch"), np.sqrt(2 * eps / (1 + eps)))
    # a particle with v∥ = 0 at θ bounces there: sin²(θ/2) = κ²
    theta = 1.2
    assert orbits.trapping_parameter(orbits.pitch_parameter(0.0, 0.1, theta), 0.1) == pytest.approx(
        np.sin(theta / 2) ** 2
    )
    with pytest.raises(ValueError, match="quantity"):
        orbits.trapping_boundary(0.1, "angle")


@pytest.mark.parametrize("epsilon, rtol", [(0.01, 1.5e-2), (0.002, 3e-3)])
def test_bounce_frequency_against_bounce_motion(epsilon, rtol):
    q = 1.5
    kappa2 = np.array([0.1, 0.5, 0.9])
    analytic = orbits.bounce_frequency(1.0, kappa2, epsilon, q, 1.0)
    periods = 2 * np.pi / analytic
    t, states = bounce_orbit(kappa2, epsilon, q, 1e3, dt=periods.min() / 2000, t_end=2.2 * periods.max())
    for j in range(len(kappa2)):
        turns = downward_crossings(t, states[:, 1, j])
        measured = 2 * np.pi / (turns[1] - turns[0])
        assert measured == pytest.approx(analytic[j], rel=rtol), kappa2[j]
        # the bounce points: sin²(θ_b/2) = κ²
        assert np.sin(states[:, 0, j].max() / 2) ** 2 == pytest.approx(kappa2[j], rel=rtol)


def test_deeply_trapped_limit():
    eps, q, r0, v = 0.004, 2.0, 3.0, 5.0
    assert orbits.bounce_frequency(v, 0.0, eps, q, r0) == pytest.approx(v * np.sqrt(eps / 2) / (q * r0))
    # a small-amplitude bounce oscillates at that frequency
    t, states = bounce_orbit(
        np.array([1e-4]),
        eps,
        q,
        1e3,
        dt=0.02,
        t_end=2.2 * 2 * np.pi * q / np.sqrt(eps / 2),
    )
    turns = downward_crossings(t, states[:, 1, 0])
    assert 2 * np.pi / (turns[1] - turns[0]) == pytest.approx(np.sqrt(eps / 2) / q, rel=2 * eps)
    # ω_b → 0 at the boundary, nan for passing particles
    assert orbits.bounce_frequency(1.0, 1.0, 0.1, 1.0, 1.0) == 0.0
    assert np.isnan(orbits.bounce_frequency(1.0, 1.5, 0.1, 1.0, 1.0))
    assert orbits.bounce_frequency(np.ones((2, 1)), np.array([0.2, 0.4, 0.6]), 0.1, 1.0, 1.0).shape == (2, 3)


def test_transit_frequency_against_quadrature():
    eps, q, r0 = 0.002, 1.3, 2.0
    kappa2 = np.array([1.2, 3.0, 20.0, 400.0])
    lam = 1 + eps - 2 * eps * kappa2
    theta = 2 * np.pi * np.arange(8192) / 8192
    # leading order in ε (what the formula states): v∥ = √(2ε(κ² − sin²(θ/2)))
    v_leading = np.sqrt(2 * eps * (kappa2[:, None] - np.sin(theta / 2) ** 2))
    period = q * r0 * np.mean(1 / v_leading, axis=-1) * 2 * np.pi
    np.testing.assert_allclose(orbits.transit_frequency(1.0, kappa2, eps, q, r0), 2 * np.pi / period, rtol=1e-6)
    # the exact v∥ in B = B₀/(1 + ε cos θ), to O(ε)
    v_exact = np.sqrt(1 - lam[:, None] / (1 + eps * np.cos(theta)))
    period = q * r0 * np.mean(1 / v_exact, axis=-1) * 2 * np.pi
    np.testing.assert_allclose(
        orbits.transit_frequency(1.0, kappa2, eps, q, r0),
        2 * np.pi / period,
        rtol=2 * eps,
    )
    # far from the boundary: |v∥|/(qR₀)
    assert orbits.transit_frequency(1.0, 1e6, eps, q, r0) == pytest.approx(np.sqrt(2 * eps * 1e6) / (q * r0), rel=1e-6)
    assert np.isnan(orbits.transit_frequency(1.0, 0.5, eps, q, r0))
    # both frequencies use the same elliptic integral at κ² → 1
    assert elliptic_k(0.999) > 4


def test_banana_width_against_drift_orbit():
    eps, q, omega = 0.002, 2.0, 1e3
    kappa2 = np.array([0.2, 0.6, 0.95])
    periods = 2 * np.pi / orbits.bounce_frequency(1.0, kappa2, eps, q, 1.0)
    t, states = bounce_orbit(kappa2, eps, q, omega, dt=periods.min() / 2000, t_end=1.05 * periods.max())
    for j in range(len(kappa2)):
        inside = t <= periods[j]
        r = states[inside, 2, j]
        width = orbits.banana_width(1 / omega, kappa2[j], eps, q)
        assert r.max() - r.min() == pytest.approx(width, rel=1e-2), kappa2[j]
    assert orbits.banana_width(0.01, 0.0, 0.1, 2.0) == 0.0
    assert orbits.banana_width(0.01, 1.0, 0.1, 2.0) == pytest.approx(2 * np.sqrt(2) * 2.0 * 0.01 / np.sqrt(0.1))
    assert np.isnan(orbits.banana_width(0.01, 1.5, 0.1, 2.0))
