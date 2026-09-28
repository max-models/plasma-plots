"""The numerics theory of plasma_plots.theory: amplification factors against closed forms and
direct integration, stability limits, phase-error orders, finite-difference and Yee dispersion
against direct application of the stencils, spline Galerkin dispersion against assembled mass
and stiffness matrices, and PIC noise against sampling."""

from math import factorial

import numpy as np
import pytest

from plasma_plots.theory import numerics as nm

ONE_STEP = [
    "explicit_euler",
    "implicit_euler",
    "implicit_midpoint",
    "rk2",
    "rk3",
    "rk4",
]
ALL = ONE_STEP + ["leapfrog", "two_step_leapfrog"]


def _order(x, err):
    """Log-log slope of |err| against x."""
    return np.polyfit(np.log(x), np.log(np.abs(err)), 1)[0]


# ---------------------------------------------------------------------------------------------
# CFL
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Time integrators
# ---------------------------------------------------------------------------------------------
def test_amplification_closed_forms():
    theta = np.linspace(-3.5, 3.5, 141)
    z = 1j * theta
    np.testing.assert_allclose(nm.amplification_factor(theta, "explicit_euler"), 1 + z)
    np.testing.assert_allclose(
        np.abs(nm.amplification_factor(theta, "forward_euler")) ** 2, 1 + theta**2
    )
    np.testing.assert_allclose(
        nm.amplification_factor(theta, "implicit_euler"), 1 / (1 - z)
    )
    for name in (
        "implicit_midpoint",
        "crank_nicolson",
        "trapezoidal",
        "discrete_gradient",
    ):
        np.testing.assert_allclose(
            np.abs(nm.amplification_factor(theta, name)), 1.0, atol=1e-15
        )
    np.testing.assert_allclose(
        nm.amplification_factor(theta, "crank_nicolson"), (1 + z / 2) / (1 - z / 2)
    )
    for name, s in (("rk2", 2), ("heun", 2), ("rk3", 3), ("ssprk3", 3), ("rk4", 4)):
        taylor = sum(z**n / factorial(n) for n in range(s + 1))
        np.testing.assert_allclose(nm.amplification_factor(theta, name), taylor)
    # leapfrog: G + 1/G = 2 − θ², |G| = 1 up to θ = 2; two-step leapfrog: G² − 2zG − 1 = 0
    g = nm.amplification_factor(theta, "stormer_verlet")
    np.testing.assert_allclose(g + 1 / g, 2 - theta**2, atol=1e-12)
    np.testing.assert_allclose(np.abs(g[np.abs(theta) <= 2]), 1.0)
    assert np.all(np.abs(g[np.abs(theta) > 2]) > 1)
    g = nm.amplification_factor(theta, "two_step_leapfrog")
    np.testing.assert_allclose(g**2 - 2 * z * g - 1, 0, atol=1e-12)
    # principal roots: close to exp(iθ) for small θ
    small = np.linspace(-0.2, 0.2, 9)
    for name in ALL:
        np.testing.assert_allclose(
            nm.amplification_factor(small, name), np.exp(1j * small), atol=0.03
        )
    assert np.ndim(nm.amplification_factor(0.3, "rk4")) == 0
    with pytest.raises(ValueError, match="unknown method"):
        nm.amplification_factor(0.1, "rk5")


def test_stability_limits():
    assert nm.stability_limit("rk4") == pytest.approx(2 * np.sqrt(2))
    assert nm.stability_limit("rk3") == pytest.approx(np.sqrt(3))
    assert nm.stability_limit("leapfrog") == 2.0
    assert nm.stability_limit("two_step_leapfrog") == 1.0
    assert nm.stability_limit("explicit_euler") == 0.0
    assert nm.stability_limit("implicit_euler") == np.inf
    theta = np.linspace(1e-3, 50, 5000)
    assert np.all(nm.amplitude_error(theta, "explicit_euler") > 0)  # never stable
    assert np.all(nm.amplitude_error(theta, "rk2") > 0)
    assert np.all(nm.amplitude_error(theta, "implicit_euler") < 0)
    np.testing.assert_allclose(
        nm.amplitude_error(theta, "implicit_midpoint"), 0, atol=1e-14
    )
    for name in ("rk3", "rk4", "leapfrog", "two_step_leapfrog"):
        limit = nm.stability_limit(name)
        inside = np.linspace(1e-3, limit * (1 - 1e-9), 1000)
        assert np.all(nm.amplitude_error(inside, name) <= 1e-12), name
        assert nm.amplitude_error(limit * 1.001, name) > 0, name
        assert nm.amplitude_error(-limit * 1.001, name) > 0, name


def test_phase_and_amplitude_error_orders():
    leading = {  # method: (coefficient, power) of the relative frequency error
        "explicit_euler": (-1 / 3, 2),
        "implicit_euler": (-1 / 3, 2),
        "implicit_midpoint": (-1 / 12, 2),
        "leapfrog": (1 / 24, 2),
        "two_step_leapfrog": (1 / 6, 2),
        "rk2": (1 / 6, 2),
        "rk3": (1 / 30, 4),
        "rk4": (-1 / 120, 4),
    }
    for name, (coefficient, power) in leading.items():
        theta = np.logspace(-3, -2, 6) if power == 2 else np.logspace(-1.5, -1, 6)
        err = nm.phase_error(theta, name)
        np.testing.assert_allclose(
            err / theta**power, coefficient, rtol=1e-2, err_msg=name
        )
        assert _order(theta, err) == pytest.approx(power, abs=0.02), name
    theta = np.logspace(-3, -2, 6)
    np.testing.assert_allclose(
        nm.amplitude_error(theta, "rk4") / theta**6, -1 / 144, rtol=1e-2
    )
    np.testing.assert_allclose(
        nm.amplitude_error(theta, "rk3") / theta**4, -1 / 24, rtol=1e-2
    )
    np.testing.assert_allclose(
        nm.amplitude_error(theta, "rk2") / theta**4, 1 / 8, rtol=1e-2
    )
    assert nm.phase_error(0.0, "rk4") == 0.0
    # even in θ (G(−θ) = conj G(θ)); arg G/θ − 1 cancels to ~1e-12 here, so compare to a few ulps of 1
    np.testing.assert_allclose(
        nm.phase_error(-theta, "rk4"),
        nm.phase_error(theta, "rk4"),
        rtol=1e-7,
        atol=4 * np.finfo(float).eps,
    )


def _integrate(name, theta, steps):
    """Integrate y' = −iωy (the package's exp(−iωt) convention) with ω dt = θ, dt = 1."""
    lam = -1j * theta
    y = np.empty(steps + 1, dtype=complex)
    y[0] = 1.0
    f = lambda v: lam * v  # noqa: E731
    if name == "two_step_leapfrog":
        y[1] = nm.amplification_factor(
            theta, name
        ).conjugate()  # start on the principal mode
        for n in range(1, steps):
            y[n + 1] = y[n - 1] + 2 * f(y[n])
        return y
    for n in range(steps):
        v = y[n]
        if name == "explicit_euler":
            y[n + 1] = v + f(v)
        elif name == "implicit_euler":
            y[n + 1] = v / (1 - lam)
        elif name == "implicit_midpoint":
            # y+ = y + f((y + y+)/2), solved by fixed-point iteration
            new = v
            for _ in range(200):
                new = v + f((v + new) / 2)
            y[n + 1] = new
        elif name == "rk2":  # Heun
            k1 = f(v)
            k2 = f(v + k1)
            y[n + 1] = v + (k1 + k2) / 2
        elif name == "rk3":  # Kutta's third-order method
            k1 = f(v)
            k2 = f(v + k1 / 2)
            k3 = f(v - k1 + 2 * k2)
            y[n + 1] = v + (k1 + 4 * k2 + k3) / 6
        elif name == "rk4":
            k1 = f(v)
            k2 = f(v + k1 / 2)
            k3 = f(v + k2 / 2)
            k4 = f(v + k3)
            y[n + 1] = v + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return y


def _measured_frequency(y, dt=1.0):
    n = np.arange(len(y))
    phase = np.unwrap(np.angle(y))
    real = -np.polyfit(n, phase, 1)[0] / dt
    imag = np.polyfit(n, np.log(np.abs(y)), 1)[0] / dt
    return real + 1j * imag


def test_numerical_frequency_against_direct_integration():
    for name in ONE_STEP + ["two_step_leapfrog"]:
        thetas = (
            [0.1, 0.4, 0.8, 1.5, 2.0]
            if name != "implicit_midpoint"
            else [0.1, 0.4, 0.8, 1.2]
        )
        if name == "two_step_leapfrog":
            thetas = [0.1, 0.4, 0.8, 0.95]
        for theta in thetas:
            y = _integrate(name, theta, 40)
            measured = _measured_frequency(y)
            # ω = θ with dt = 1
            predicted = nm.numerical_frequency(theta, 1.0, name)
            assert measured == pytest.approx(predicted, abs=1e-9), (name, theta)
    # dt scaling: ω_num(ω, dt) = ω_num(ω dt, 1)/dt
    assert nm.numerical_frequency(2.0, 0.25, "rk4") == pytest.approx(
        nm.numerical_frequency(0.5, 1.0, "rk4") / 0.25
    )


def test_verlet_frequency_against_direct_integration():
    """Kick–drift–kick Störmer–Verlet on x'' = −ω²x: x_{n+1} + x_{n−1} = 2 cos(ω_num dt) x_n."""
    dt = 0.1
    for omega in [1.0, 5.0, 15.0, 19.0, 21.0, 30.0]:
        x, v = 1.0, 0.3
        xs = [x]
        for _ in range(12):
            v -= dt / 2 * omega**2 * x
            x += dt * v
            v -= dt / 2 * omega**2 * x
            xs.append(x)
        xs = np.array(xs)
        c = np.linalg.lstsq(xs[1:-1, None], (xs[2:] + xs[:-2]) / 2, rcond=None)[0][0]
        predicted = nm.numerical_frequency(omega, dt, "leapfrog")
        assert np.cos(predicted * dt) == pytest.approx(c, rel=1e-9), omega
        if omega * dt > 2:
            assert predicted.real == pytest.approx(np.pi / dt)
            # the growth factor per step is the larger root
            growth = np.abs(xs[-1] / xs[-2])
            assert growth == pytest.approx(np.exp(predicted.imag * dt), rel=1e-2)


# ---------------------------------------------------------------------------------------------
# Spatial discretizations
# ---------------------------------------------------------------------------------------------
STENCILS_FIRST = {
    2: {1: 1 / 2},
    4: {1: 2 / 3, 2: -1 / 12},
    6: {1: 3 / 4, 2: -3 / 20, 3: 1 / 60},
}
STENCILS_SECOND = {
    2: {0: -2, 1: 1},
    4: {0: -5 / 2, 1: 4 / 3, 2: -1 / 12},
    6: {0: -49 / 18, 1: 3 / 2, 2: -3 / 20, 3: 1 / 90},
}


def _bspline(p, x):
    """Cardinal B-spline of degree p on [0, p + 1] by the Cox–de Boor recursion."""
    x = np.asarray(x, dtype=float)
    if p == 0:
        return ((x >= 0) & (x < 1)).astype(float)
    return (x * _bspline(p - 1, x) + (p + 1 - x) * _bspline(p - 1, x - 1)) / p


def _dbspline(p, x):
    return _bspline(p - 1, x) - _bspline(p - 1, x - 1)


def _assemble(p, n, h):
    """Periodic mass and stiffness matrices of degree-p B-splines on n cells, by Gauss quadrature."""
    t, w = np.polynomial.legendre.leggauss(p + 2)
    t, w = (t + 1) / 2, w / 2
    mass = np.zeros((n, n))
    stiff = np.zeros((n, n))
    for cell in range(n):
        for li in range(p + 1):  # basis i = cell − li is evaluated at li + t
            i = (cell - li) % n
            for lj in range(p + 1):
                j = (cell - lj) % n
                mass[i, j] += h * np.sum(w * _bspline(p, li + t) * _bspline(p, lj + t))
                stiff[i, j] += (
                    np.sum(w * _dbspline(p, li + t) * _dbspline(p, lj + t)) / h
                )
    return mass, stiff


def test_spline_galerkin_dispersion_against_assembled_matrices():
    n, length, c = 24, 2.0, 1.5
    h = length / n
    k = 2 * np.pi * np.arange(n) / length
    for p in (1, 2, 3, 4):
        mass, stiff = _assemble(p, n, h)
        assert mass.sum() == pytest.approx(length)  # partition of unity
        eig = np.linalg.eigvals(np.linalg.solve(mass, stiff)).real
        measured = np.sort(c * np.sqrt(np.clip(eig, 0, None)))
        predicted = np.sort(nm.spline_galerkin_dispersion(k, h, p, c))
        np.testing.assert_allclose(measured, predicted, rtol=1e-10, atol=1e-6)
    # linear finite elements in closed form
    theta = np.linspace(0.01, np.pi, 20)
    np.testing.assert_allclose(
        nm.spline_galerkin_dispersion(theta, 1.0, 1) ** 2,
        6 * (1 - np.cos(theta)) / (2 + np.cos(theta)),
    )


def test_spline_galerkin_convergence_order():
    """ω/(ck) − 1 ≈ |B_2p| θ^2p / (2 (2p)!), order 2p."""
    ranges = {1: (-2, -1.5), 2: (-1.5, -1.0), 3: (-1.3, -0.9), 4: (-0.9, -0.6)}
    coefficients = {1: 1 / 24, 2: 1 / 1440, 3: 1 / 60480, 4: 1 / 2419200}
    for p, (lo, hi) in ranges.items():
        theta = np.logspace(lo, hi, 5)
        err = nm.spline_galerkin_dispersion(theta, 1.0, p) / theta - 1
        assert np.all(err > 0)
        assert _order(theta, err) == pytest.approx(2 * p, abs=0.05 * p), p
        np.testing.assert_allclose(
            err[0] / theta[0] ** (2 * p), coefficients[p], rtol=2e-2 if p < 4 else 5e-2
        )
    with pytest.raises(ValueError, match="degree"):
        nm.spline_galerkin_dispersion(1.0, 0.1, 0)
    with pytest.raises(ValueError, match="degree"):
        nm.spline_galerkin_dispersion(1.0, 0.1, 1.5)


# ---------------------------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------------------------


def test_scipy_eigenvalues_agree():
    linalg = pytest.importorskip("scipy.linalg")
    mass, stiff = _assemble(2, 16, 1 / 16)
    eig = np.sort(
        np.sqrt(np.clip(linalg.eigh(stiff, mass, eigvals_only=True), 0, None))
    )
    np.testing.assert_allclose(
        eig,
        np.sort(nm.spline_galerkin_dispersion(2 * np.pi * np.arange(16), 1 / 16, 2)),
        atol=1e-8,
    )
