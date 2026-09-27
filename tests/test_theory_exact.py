"""The exact solutions of struphy_plots.theory.exact: published benchmark values (Toro's Riemann
problems), conservation laws, and the differential equations checked by finite differences."""

import numpy as np
import pytest

from struphy_plots.theory.exact import (
    advected,
    caustic_time,
    dalembert,
    dam_break,
    heat_kernel,
    pressureless,
    pressureless_eulerian,
    riemann_euler,
    sod_shock_tube,
    star_state,
)

H = 1e-4  # finite-difference step: central differences are accurate to about H² ~ 1e-8


def d(f, h=H):
    """Central difference of f(offset) at offset 0."""
    return (f(h) - f(-h)) / (2 * h)


def d2(f, h=1e-3):
    """Second central difference of f(offset) at offset 0."""
    return (f(h) - 2 * f(0.0) + f(-h)) / h**2


# ---------------------------------------------------------------------------------------------
# Riemann problem
# ---------------------------------------------------------------------------------------------
# Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, Tables 4.1 and 4.2 (γ = 1.4):
# left state, right state, p*, u*, ρ*L, ρ*R
TORO = [
    ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), 0.30313, 0.92745, 0.42632, 0.26557),
    ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4), 0.00189, 0.00000, 0.02185, 0.02185),
    ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 460.894, 19.5975, 0.57506, 5.99924),
    ((1.0, 0.0, 0.01), (1.0, 0.0, 100.0), 46.0950, -6.19633, 5.99242, 0.57511),
    ((5.99924, 19.5975, 460.894), (5.99242, -6.19633, 46.0950), 1691.64, 8.68975, 14.2823, 31.0426),
]
TORO_WAVES = [("rarefaction", "shock"), ("rarefaction", "rarefaction"), ("rarefaction", "shock"),
              ("shock", "rarefaction"), ("shock", "shock")]


@pytest.mark.parametrize("case, waves", list(zip(TORO, TORO_WAVES)))
def test_star_states_match_toro(case, waves):
    left, right, p, u, rho_l, rho_r = case
    s = star_state(left, right)
    # the table has five or six significant digits
    assert s.pressure == pytest.approx(p, rel=5e-5, abs=5e-6)
    assert s.velocity == pytest.approx(u, rel=5e-5, abs=5e-6)
    assert s.density_left == pytest.approx(rho_l, rel=5e-5, abs=5e-6)
    assert s.density_right == pytest.approx(rho_r, rel=5e-5, abs=5e-6)
    assert (s.left_wave, s.right_wave) == waves
    assert not s.vacuum


def test_star_pressure_agrees_with_scipy():
    optimize = pytest.importorskip("scipy.optimize")
    gamma = 5 / 3
    rng = np.random.default_rng(1)
    for _ in range(50):
        left = (rng.uniform(0.1, 10), rng.uniform(-3, 3), rng.uniform(0.01, 100))
        right = (rng.uniform(0.1, 10), rng.uniform(-3, 3), rng.uniform(0.01, 100))

        def f_k(p, rho, pk):
            a = np.sqrt(gamma * pk / rho)
            if p > pk:
                return (p - pk) * np.sqrt(2 / ((gamma + 1) * rho) / (p + (gamma - 1) / (gamma + 1) * pk))
            return 2 * a / (gamma - 1) * ((p / pk) ** ((gamma - 1) / (2 * gamma)) - 1)

        def f(p, left=left, right=right):
            return f_k(p, left[0], left[2]) + f_k(p, right[0], right[2]) + right[1] - left[1]

        s = star_state(left, right, gamma)
        if s.vacuum:
            assert f(1e-300) >= 0
            continue
        assert s.pressure == pytest.approx(optimize.brentq(f, 1e-300, 1e6, xtol=1e-300, rtol=1e-15), rel=1e-12)


def _flux(rho, u, p, gamma):
    energy = 0.5 * rho * u**2 + p / (gamma - 1)
    return np.array([rho * u, rho * u**2 + p, u * (energy + p)])


def _conserved(w):
    return np.array([w.density, w.momentum, w.energy])


def _check_conservation(left, right, gamma, t, a=1.0, n=400_001):
    """∫U(t) dx = ∫U(0) dx + t [F(W_L) − F(W_R)] over [−a, a] containing all waves."""
    x = np.linspace(-a, a, n)
    w = riemann_euler(x, t, left, right, gamma)
    # the domain contains all the waves: the boundary states are still the initial ones
    np.testing.assert_allclose(_conserved(w)[:, [0, -1]], _conserved(riemann_euler([-a, a], 0.0, left, right, gamma)),
                               atol=1e-14)
    conserved = _conserved(w)
    integral = np.trapezoid(conserved, x, axis=1)
    u_left, u_right = (_conserved(riemann_euler(-a, 0.0, left, right, gamma)),
                       _conserved(riemann_euler(a, 0.0, left, right, gamma)))
    expected = a * (u_left + u_right) + t * (_flux(*left, gamma) - _flux(*right, gamma))
    # the trapezoidal rule errs by up to a cell width times the jump at each discontinuity
    bound = (x[1] - x[0]) * np.sum(np.abs(np.diff(conserved, axis=1)), axis=1) / 2
    scale = a * (np.abs(u_left) + np.abs(u_right))
    assert np.all(np.abs(integral - expected) <= bound + 1e-10 * scale), (integral - expected, bound)


@pytest.mark.parametrize("case, t", list(zip(TORO, [0.25, 0.15, 0.012, 0.035, 0.035])))
def test_riemann_solution_conserves_mass_momentum_energy(case, t):
    _check_conservation(case[0], case[1], 1.4, t)


def test_riemann_solution_values_and_structure():
    # Sod at t = 0.2: the plateaus of Toro's test 1
    w = sod_shock_tube(np.array([0.0, 0.55, 0.8, 1.0]), 0.2)
    np.testing.assert_allclose(w.density, [1.0, 0.42632, 0.26557, 0.125], atol=5e-6)
    np.testing.assert_allclose(w.pressure, [1.0, 0.30313, 0.30313, 0.1], atol=5e-6)
    np.testing.assert_allclose(w.internal_energy, w.pressure / (0.4 * w.density))
    # broadcasting of x and t, and scalars stay scalars
    assert riemann_euler(np.zeros((3, 1)), np.array([0.1, 0.2]), (1, 0, 1), (0.125, 0, 0.1)).density.shape == (3, 2)
    assert np.ndim(sod_shock_tube(0.3, 0.1).pressure) == 0
    # t = 0 gives the initial data
    w = riemann_euler([-1e-9, 1e-9], 0.0, (1, 0.5, 1), (0.125, -0.5, 0.1), x0=0.0)
    np.testing.assert_allclose([w.density, w.velocity, w.pressure], [[1, 0.125], [0.5, -0.5], [1, 0.1]])
    # inside the rarefaction: isentropic and the Riemann invariant u + 2a/(γ − 1) is constant
    x = np.linspace(-0.23, -0.02, 20)
    w = sod_shock_tube(x + 0.5, 0.2)
    np.testing.assert_allclose(w.pressure / w.density**1.4, 1.0, rtol=1e-12)
    np.testing.assert_allclose(w.velocity + 2 * w.sound_speed / 0.4, 2 * np.sqrt(1.4) / 0.4, rtol=1e-12)
    # the mirror image of a problem is the mirrored solution
    left, right = (1.0, 0.3, 2.0), (0.4, -0.7, 0.5)
    x = np.linspace(-1, 1, 101)
    a, b = riemann_euler(x, 0.3, left, right, 5 / 3), riemann_euler(-x, 0.3, right[:1] + (0.7, 0.5), (1.0, -0.3, 2.0),
                                                                    5 / 3)
    np.testing.assert_allclose(a.density, b.density, rtol=1e-12)
    np.testing.assert_allclose(a.velocity, -b.velocity, rtol=1e-12, atol=1e-14)
    # identical states: nothing happens
    w = riemann_euler(x, 1.0, (1.0, 0.2, 1.0), (1.0, 0.2, 1.0))
    np.testing.assert_allclose([w.density, w.velocity, w.pressure], [np.ones(101), 0.2 * np.ones(101), np.ones(101)])


@pytest.mark.parametrize("left, right", [
    ((1.0, 0.0, 1.0), (0.0, 0.0, 0.0)),        # expansion into vacuum to the right
    ((0.0, 0.0, 0.0), (1.0, 0.5, 1.0)),        # expansion into vacuum to the left
    ((1.0, -4.0, 0.4), (1.0, 4.0, 0.4)),       # vacuum generated between two rarefactions
    ((1.0, -6.0, 1.0), (0.5, 6.0, 0.3)),       # the same, asymmetric
])
def test_vacuum_cases(left, right):
    gamma = 1.4
    s = star_state(left, right, gamma)
    assert s.vacuum and s.pressure == 0
    t = 0.1
    _check_conservation(left, right, gamma, t, a=2.0)
    x = np.linspace(-2, 2, 4001)
    w = riemann_euler(x, t, left, right, gamma)
    empty = w.density == 0
    assert empty.any()
    assert np.all(np.isnan(w.velocity[empty])) and np.all(w.pressure[empty] == 0)
    assert not np.isnan(w.velocity[~empty]).any()
    # the fronts move at u_L + 2a_L/(γ − 1) and u_R − 2a_R/(γ − 1)
    fronts = []
    if left[0] > 0:
        fronts.append(left[1] + 2 * np.sqrt(gamma * left[2] / left[0]) / (gamma - 1))
        # the Riemann invariant u + 2a/(γ − 1) is carried through the left fan up to the front
        fan = (~empty) & (x < fronts[-1] * t) & (x > (left[1] - np.sqrt(gamma * left[2] / left[0])) * t)
        np.testing.assert_allclose((w.velocity + 2 * w.sound_speed / (gamma - 1))[fan], fronts[-1], rtol=1e-12)
        assert riemann_euler(fronts[-1] * t - 1e-9, t, left, right, gamma).density < 1e-12
        assert riemann_euler(fronts[-1] * t + 1e-9, t, left, right, gamma).density == 0
    if right[0] > 0:
        front = right[1] - 2 * np.sqrt(gamma * right[2] / right[0]) / (gamma - 1)
        assert riemann_euler(front * t - 1e-9, t, left, right, gamma).density == 0
        assert riemann_euler(front * t + 1e-9, t, left, right, gamma).density < 1e-12
    # the density approaches zero continuously at the vacuum boundary
    assert np.max(np.abs(np.diff(w.density))) < 0.05


def test_vacuum_limit_is_continuous():
    # approaching the critical velocity jump, p* → 0 and the solution approaches the vacuum solution
    gamma = 1.4
    critical = 2 * 2 * np.sqrt(gamma * 0.4) / (gamma - 1)
    s = star_state((1.0, -0.5 * critical + 1e-4, 0.4), (1.0, 0.5 * critical - 1e-4, 0.4), gamma)
    assert not s.vacuum and 0 < s.pressure < 1e-12
    x = np.linspace(-1, 1, 201)
    near = riemann_euler(x, 0.1, (1.0, -0.5 * critical + 1e-6, 0.4), (1.0, 0.5 * critical - 1e-6, 0.4), gamma)
    vac = riemann_euler(x, 0.1, (1.0, -0.5 * critical - 1e-6, 0.4), (1.0, 0.5 * critical + 1e-6, 0.4), gamma)
    np.testing.assert_allclose(near.density, vac.density, atol=1e-5)


def test_riemann_errors():
    with pytest.raises(ValueError, match="gamma"):
        star_state((1, 0, 1), (1, 0, 1), gamma=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        star_state((1, 0, -1), (1, 0, 1))
    with pytest.raises(ValueError, match="vacuum"):
        star_state((0, 0, 1), (1, 0, 1))
    with pytest.raises(ValueError, match="t must"):
        sod_shock_tube(0.5, -0.1)


# ---------------------------------------------------------------------------------------------
# Dam break
# ---------------------------------------------------------------------------------------------
def test_dam_break_conserves_volume_and_momentum():
    h0, g, t = 2.0, 9.81, 0.3
    c0 = np.sqrt(g * h0)
    a = 3 * c0 * t
    x = np.linspace(-a, a, 600_001)
    w = dam_break(x, t, depth=h0, gravity=g)
    volume = np.trapezoid(w.depth, x)
    assert volume == pytest.approx(a * h0, rel=1e-6)
    # d/dt ∫ hu dx = [g h²/2] flux difference at the (still) boundaries
    momentum = np.trapezoid(w.discharge, x)
    assert momentum == pytest.approx(0.5 * g * h0**2 * t, rel=1e-5)
    # the Riemann invariant u + 2√(gh) = 2c0 through the rarefaction
    behind = (x > -c0 * t) & (w.depth > 0)
    np.testing.assert_allclose((w.velocity + 2 * np.sqrt(g * w.depth))[behind], 2 * c0, rtol=1e-12)


def test_ritter_front_and_profile():
    h0, g = 1.5, 2.0
    c0 = np.sqrt(g * h0)
    for t in (0.5, 1.0, 2.0):
        front = 2 * c0 * t
        assert dam_break(front - 1e-9, t, depth=h0, gravity=g).depth > 0
        assert dam_break(front + 1e-9, t, depth=h0, gravity=g).depth == 0
        assert np.isnan(dam_break(front + 1e-9, t, depth=h0, gravity=g).velocity)
        assert dam_break(-c0 * t - 1e-9, t, depth=h0, gravity=g).depth == h0
    # the depth at the dam site stays 4h0/9, the velocity 2c0/3
    w = dam_break(0.0, [0.1, 1.0, 10.0], depth=h0, gravity=g)
    np.testing.assert_allclose(w.depth, 4 * h0 / 9)
    np.testing.assert_allclose(w.velocity, 2 * c0 / 3)
    with pytest.raises(ValueError):
        dam_break(0.0, 1.0, depth=0.0)


# ---------------------------------------------------------------------------------------------
# Diffusion, advection, waves
# ---------------------------------------------------------------------------------------------
def test_heat_kernel_solves_the_heat_equation_and_conserves_mass():
    D, t = 0.3, 0.7
    x = np.linspace(-3, 3, 13)
    for width in (0.0, 0.4):
        u_t = d(lambda h: heat_kernel(x, t + h, D, width=width, center=0.2, mass=2.0))
        u_xx = d2(lambda h: heat_kernel(x + h, t, D, width=width, center=0.2, mass=2.0))
        np.testing.assert_allclose(u_t, D * u_xx, atol=2e-6)
        grid = np.linspace(-20, 20, 40001)
        u = heat_kernel(grid, t, D, width=width, center=0.2, mass=2.0)
        assert np.trapezoid(u, grid) == pytest.approx(2.0, rel=1e-10)
        variance = np.trapezoid(u * (grid - 0.2) ** 2, grid) / 2.0
        assert variance == pytest.approx(width**2 + 2 * D * t, rel=1e-8)
    # two dimensions
    X, Y = np.meshgrid(np.linspace(-2, 2, 5), np.linspace(-1, 1, 5))

    def u(dx=0.0, dy=0.0, dt=0.0):
        return heat_kernel((X + dx, Y + dy), t + dt, D, width=0.3, center=(0.1, -0.2))

    lap = d2(lambda h: u(dx=h)) + d2(lambda h: u(dy=h))
    np.testing.assert_allclose(d(lambda h: u(dt=h)), D * lap, atol=2e-6)
    grid = np.linspace(-12, 12, 1201)
    GX, GY = np.meshgrid(grid, grid)
    total = np.trapezoid(np.trapezoid(heat_kernel((GX, GY), t, D, width=0.3), grid), grid)
    assert total == pytest.approx(1.0, rel=1e-10)
    # a point source at t = 0
    assert np.isinf(heat_kernel(0.0, 0.0, D)) and heat_kernel(1.0, 0.0, D) == 0
    with pytest.raises(ValueError):
        heat_kernel(0.0, -1.0, D)


def test_advected():
    def profile(x):
        return np.exp(-(x**2) / 0.1)

    x = np.linspace(-1, 1, 11)
    # ∂u/∂t + v ∂u/∂x = 0
    v = 0.7
    u_t = d(lambda h: advected(profile, x, 0.3 + h, v))
    u_x = d(lambda h: advected(profile, x + h, 0.3, v))
    np.testing.assert_allclose(u_t + v * u_x, 0, atol=1e-7)
    # periodic: after one period the profile is back; bounds as (a, b)
    np.testing.assert_allclose(advected(profile, x, 2.0 / v, v, period=(-1.0, 1.0)), profile(x), atol=1e-12)
    np.testing.assert_allclose(advected(np.sin, x, 0.5, 2 * np.pi, period=2 * np.pi), np.sin(x - np.pi), atol=1e-12)


def test_dalembert_solves_the_wave_equation():
    c = 1.5

    def u0(x):
        return np.exp(-(x**2) / 0.2)

    def v0(x):
        return np.sin(3 * x) * np.exp(-(x**2))

    x = np.linspace(-2, 2, 9)
    t = 0.4

    def u(dx=0.0, dt=0.0):
        return dalembert(x + dx, t + dt, u0, c, initial_rate=v0)

    np.testing.assert_allclose(d2(lambda h: u(dt=h)), c**2 * d2(lambda h: u(dx=h)), rtol=5e-5, atol=1e-5)
    # the initial conditions
    np.testing.assert_allclose(dalembert(x, 0.0, u0, c, v0), u0(x))
    np.testing.assert_allclose(d(lambda h: dalembert(x, h, u0, c, v0)), v0(x), atol=1e-7)
    # an initial rate cos(kx): u = cos(kx) sin(kct)/(kc)
    k = 2.0
    np.testing.assert_allclose(dalembert(x, t, np.zeros_like, c, lambda s: np.cos(k * s)),
                               np.cos(k * x) * np.sin(k * c * t) / (k * c), rtol=1e-13, atol=1e-15)
    # the standing wave A sin(kx) cos(kct) is the solution for u0 = A sin(kx) at rest
    np.testing.assert_allclose(dalembert(x, t, lambda s: 0.5 * np.sin(k * s), c),
                               0.5 * np.sin(k * x) * np.cos(k * c * t), atol=1e-15)
    with pytest.raises(ValueError):
        dalembert(x, t, u0, 0.0)


# ---------------------------------------------------------------------------------------------
# Pressureless flow
# ---------------------------------------------------------------------------------------------
def test_pressureless_flow_conserves_mass_until_the_caustic():
    def v0(q):
        return -0.5 * np.sin(q)

    q = np.linspace(-np.pi, np.pi, 2001)
    assert caustic_time(v0, q) == pytest.approx(2.0, rel=1e-9)
    assert caustic_time(np.cos, np.linspace(-np.pi / 2, 0, 11)) == np.inf  # increasing here
    for t in (0.0, 0.5, 1.5, 1.9):
        # Lagrangian: ρ dx/dq = ρ0
        flow = pressureless(q, t, v0, density=lambda q: 1 + 0.2 * np.cos(q))
        dxdq = np.gradient(flow.position, q)
        np.testing.assert_allclose(flow.density * dxdq, 1 + 0.2 * np.cos(q), rtol=2e-4)
        # Eulerian: the mass in one period is conserved
        x = np.linspace(-np.pi, np.pi, 20001)
        eulerian = pressureless_eulerian(x, t, v0)
        assert np.trapezoid(eulerian.density, x) == pytest.approx(2 * np.pi, rel=1e-4 if t > 1.6 else 1e-7)
        # the inverse map is consistent with the forward one
        back = pressureless_eulerian(flow.position, t, v0)
        np.testing.assert_allclose(back.velocity, flow.velocity, atol=1e-12)
    # the density grows without bound at the caustic
    assert pressureless(0.0, 2.0, v0).density > 1e9
    assert pressureless(0.0, 2.0, v0, velocity_derivative=lambda q: -0.5 * np.cos(q)).density == np.inf
    # the analytic derivative gives the same
    np.testing.assert_allclose(pressureless(q, 1.0, v0, velocity_derivative=lambda q: -0.5 * np.cos(q)).density,
                               pressureless(q, 1.0, v0).density, rtol=1e-8)


def test_pressureless_eulerian_on_an_unbounded_velocity():
    # v0 = −q: every element reaches x = 0 at t = 1; before, ρ = 1/(1 − t) and v = −x/(1 − t)
    x = np.linspace(-5, 5, 11)
    flow = pressureless_eulerian(x, 0.6, lambda q: -q)
    np.testing.assert_allclose(flow.density, 1 / 0.4)
    np.testing.assert_allclose(flow.velocity, -x / 0.4, atol=1e-12)


# ---------------------------------------------------------------------------------------------
# Decaying modes
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Incompressible flows
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Cavity fields
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Manufactured solutions
# ---------------------------------------------------------------------------------------------
