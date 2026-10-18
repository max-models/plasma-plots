"""The equilibria of plasma_plots.theory.equilibria: the force balance ∇p = J × B and ∇ · B = 0
checked by finite differences of the returned fields, the Bennett relation, the shape of the
Solov'ev boundary, q on the axis against a traced field line, and Bessel function values.
"""

import numpy as np
import pytest

from plasma_plots.theory import equilibria as eq
from plasma_plots.theory.equilibria import (
    bennett_pinch,
    harris_drift_speed,
    harris_sheet,
    lundquist_field,
    solovev,
    solovev_axis,
    solovev_boundary,
    solovev_coefficients,
    solovev_safety_factor_axis,
    taylor_state,
)
from plasma_plots.theory.parameters import elementary_charge as E
from plasma_plots.theory.parameters import vacuum_permeability as MU0
from plasma_plots.theory.special import bessel_j

H = 1e-5


def d(f, x, h=H):
    return (f(x + h) - f(x - h)) / (2 * h)


# ---------------------------------------------------------------------------------------------
# Bessel functions
# ---------------------------------------------------------------------------------------------
def test_bessel_tabulated_values():
    # Abramowitz and Stegun table 9.1 and the zeros of J0 and J1
    assert bessel_j(0, 1.0) == pytest.approx(0.7651976866, abs=1e-10)
    assert bessel_j(1, 1.0) == pytest.approx(0.4400505857, abs=1e-10)
    assert bessel_j(2, 1.0) == pytest.approx(0.1149034849, abs=1e-10)
    assert bessel_j(1, 50.0) == pytest.approx(-0.09751182, abs=1e-8)
    assert abs(bessel_j(0, 2.404825557695773)) < 1e-14
    assert abs(bessel_j(1, 3.831705970207512)) < 1e-14
    assert bessel_j(0, 0.0) == 1.0 and abs(bessel_j(1, 0.0)) < 1e-15


def test_bessel_recurrence_and_derivative():
    x = np.linspace(0.1, 30, 50)
    assert np.allclose(bessel_j(0, x) + bessel_j(2, x), 2 / x * bessel_j(1, x), atol=1e-13)
    assert np.allclose(d(lambda t: bessel_j(0, t), x), -bessel_j(1, x), atol=1e-9)
    with pytest.raises(ValueError):
        bessel_j(0.5, 1.0)


# ---------------------------------------------------------------------------------------------
# Harris sheet
# ---------------------------------------------------------------------------------------------
def test_harris_force_balance_and_current():
    z = np.linspace(-3, 3, 61)
    kw = dict(b0=0.02, thickness=0.7, temperature_electron=3.0, temperature_ion=7.0, background_density=1e17)
    s = harris_sheet(z, **kw)
    # dp/dz = (J × B)_z = −J_y B_x
    dp = d(lambda t: harris_sheet(t, **kw).pressure, z, 1e-4)
    assert np.allclose(dp, -s.current_density * s.b_x, rtol=1e-6, atol=1e-6 * np.abs(dp).max())
    # μ₀ J_y = dB_x/dz and B_x = −dA_y/dz
    assert np.allclose(d(lambda t: harris_sheet(t, **kw).b_x, z, 1e-4), MU0 * s.current_density, rtol=1e-6)
    assert np.allclose(-d(lambda t: harris_sheet(t, **kw).vector_potential, z, 1e-4), s.b_x, atol=1e-9)
    total = s.pressure + s.b_x**2 / (2 * MU0)
    assert np.ptp(total) < 1e-12 * total[0]


def test_harris_drifts_carry_the_current():
    kw = dict(b0=0.02, thickness=0.7, temperature_electron=3.0, temperature_ion=7.0)
    s = harris_sheet(0.0, **kw)
    drift = harris_drift_speed(3.0, 0.02, 0.7) + harris_drift_speed(7.0, 0.02, 0.7)
    assert s.peak_density * E * drift == pytest.approx(s.current_density)
    assert harris_sheet(1e4, **kw).vector_potential == pytest.approx(-0.02 * 0.7 * (1e4 / 0.7 - np.log(2)))
    with pytest.raises(ValueError):
        harris_sheet(0.0, 1.0, -1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------------------------
# Bennett pinch
# ---------------------------------------------------------------------------------------------
def test_bennett_force_balance_relation_and_current():
    r = np.linspace(0.05, 6, 80) * 1e-3
    args = (1e20, 1e-3, 40.0, 60.0)
    s = bennett_pinch(r, *args)
    dp = d(lambda t: bennett_pinch(t, *args).pressure, r, 1e-8)
    assert np.allclose(dp, -s.current_density * s.b_theta, rtol=1e-6)
    # μ₀ J_z = (1/r) d(r B_θ)/dr
    curl = d(lambda t: t * bennett_pinch(t, *args).b_theta, r, 1e-8) / r
    assert np.allclose(curl, MU0 * s.current_density, rtol=1e-6)
    # Bennett relation, N from integrating n over the plane (substitution r = a tan φ)
    phi = np.linspace(0, np.pi / 2, 200001)
    integrand = (
        bennett_pinch(1e-3 * np.tan(phi), *args).density * 2 * np.pi * 1e-3 * np.tan(phi) * 1e-3 / np.cos(phi) ** 2
    )
    n_line = np.trapezoid(integrand, phi)
    assert n_line == pytest.approx(s.line_density, rel=1e-6)
    assert MU0 * s.total_current**2 == pytest.approx(8 * np.pi * n_line * E * 100.0, rel=1e-6)
    # the enclosed current approaches I and B_θ = μ₀ I_enclosed/(2π r)
    assert s.enclosed_current[-1] < s.total_current
    assert np.allclose(s.b_theta, MU0 * s.enclosed_current / (2 * np.pi * r))
    with pytest.raises(ValueError):
        bennett_pinch(1.0, 1e20, 0.0, 1.0, 1.0)


# ---------------------------------------------------------------------------------------------
# Solov'ev equilibrium
# ---------------------------------------------------------------------------------------------
SHAPES = [
    (0.32, 1.7, 0.33, -0.155),  # ITER-like
    (0.3, 1.5, 0.2, 0.1),
    (0.78, 2.0, 0.35, 0.0),  # spherical-tokamak-like
    (0.2, 1.0, 0.0, 0.5),  # circular
    (0.25, 1.4, -0.3, 0.3),  # negative triangularity
]


@pytest.mark.parametrize("shape", SHAPES)
def test_solovev_basis_solves_grad_shafranov(shape):
    """Δ*ψ̂ = A + (1 − A) x² for the full ψ̂, and Δ*ψᵢ = 0 for each homogeneous solution."""
    eps, kappa, delta, a = shape
    x, y = np.meshgrid(np.linspace(0.6, 1.4, 15), np.linspace(-0.4, 0.4, 15), indexing="ij")
    h = 2e-3

    def lap(f):
        fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / h**2
        fx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / h**2
        return fxx - fx / x + fyy

    for i in range(7):  # the central differences are O(h²) accurate
        f = lambda a_, b_: eq._basis(a_, b_)[i]  # noqa: E731
        assert np.abs(lap(f)).max() < 1e-3 * max(np.abs(f(x, y)).max(), 1.0)
    c = solovev_coefficients(*shape)
    psi = lambda a_, b_: eq._particular(a_, b_, a) + np.tensordot(c, eq._basis(a_, b_), axes=1)  # noqa: E731
    assert np.allclose(lap(psi), a + (1 - a) * x**2, atol=1e-4)


def test_solovev_basis_derivatives():
    x, y = np.meshgrid(np.linspace(0.6, 1.4, 9), np.linspace(-0.4, 0.4, 9), indexing="ij")
    h = 1e-5
    for dx, dy in [(1, 0), (0, 1), (2, 0), (0, 2)]:
        lower = (dx - 1, dy) if dx else (dx, dy - 1)
        fx, fy = (h, 0.0) if dx else (0.0, h)
        fd = (eq._basis(x + fx, y + fy, *lower) - eq._basis(x - fx, y - fy, *lower)) / (2 * h)
        assert np.allclose(eq._basis(x, y, dx, dy), fd, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
def test_solovev_boundary_shape(shape):
    eps, kappa, delta, a = shape
    # exact at the three defining points
    for tau in (0.0, np.pi / 2, np.pi):
        r, z = solovev_boundary(eps, kappa, delta, tau)
        assert abs(solovev(r, z, 1.0, *shape, 0.1, 1.0).psi_normalized) < 1e-12
    # the top point is (1 − δε, κε) and there ∂_R ψ = 0, so B_Z = 0
    r, z = solovev_boundary(eps, kappa, delta, np.pi / 2)
    assert (r, z) == pytest.approx((1 - delta * eps, kappa * eps))
    assert abs(solovev(r, z, 1.0, *shape, 0.1, 1.0).b_z) < 1e-10
    # elsewhere the fit is to the boundary only approximately
    tau = np.linspace(0, 2 * np.pi, 361)
    r, z = solovev_boundary(eps, kappa, delta, tau)
    psi = solovev(r, z, 1.0, *shape, 0.1, 1.0).psi_normalized
    axis = solovev(solovev_axis(*shape), 0.0, 1.0, *shape, 0.1, 1.0).psi_normalized
    assert np.abs(psi).max() < 1e-3 * abs(axis)
    assert axis < 0
    # symmetric in Z
    s1, s2 = solovev(1.1, 0.2, 1.0, *shape, 0.1, 1.0), solovev(1.1, -0.2, 1.0, *shape, 0.1, 1.0)
    assert s1.psi == pytest.approx(s2.psi) and s1.b_r == pytest.approx(-s2.b_r) and s1.b_z == pytest.approx(s2.b_z)


@pytest.mark.parametrize("shape", SHAPES)
def test_solovev_force_balance_and_divergence(shape):
    """∇p = J × B with J = ∇ × B/μ₀ by finite differences of the returned B, and ∇ · B = 0."""
    eps, kappa, delta, a = shape
    r0, psi0, b0 = 1.7, 0.4, 2.5

    def f(r, z):
        return solovev(r, z, r0, *shape, psi0, b0)

    x = r0 * np.linspace(1 - 0.9 * eps, 1 + 0.9 * eps, 31)
    y = r0 * np.linspace(-0.5 * kappa * eps, 0.5 * kappa * eps, 31)
    r, z = np.meshgrid(x, y, indexing="ij")
    h = 1e-5

    def dr(g):
        return (g(r + h, z) - g(r - h, z)) / (2 * h)

    def dz(g):
        return (g(r, z + h) - g(r, z - h)) / (2 * h)

    b = f(r, z)
    j_phi = (dz(lambda *p: f(*p).b_r) - dr(lambda *p: f(*p).b_z)) / MU0
    j_r = -dz(lambda *p: f(*p).b_phi) / MU0
    j_z = dr(lambda *p: p[0] * f(*p).b_phi) / (r * MU0)
    force_r = j_phi * b.b_z - j_z * b.b_phi
    force_z = j_r * b.b_phi - j_phi * b.b_r
    scale = np.abs(j_phi * b.b_z).max()
    assert np.abs(force_r - dr(lambda *p: f(*p).pressure)).max() < 1e-6 * scale
    assert np.abs(force_z - dz(lambda *p: f(*p).pressure)).max() < 1e-6 * scale
    div = dr(lambda *p: p[0] * f(*p).b_r) / r + dz(lambda *p: f(*p).b_z)
    assert np.abs(div).max() < 1e-8 * np.abs(b.b_z).max() / (eps * r0)
    # the closed-form toroidal current density matches the curl of B
    assert np.allclose(j_phi, b.current_density_phi, rtol=1e-6, atol=1e-6 * np.abs(j_phi).max())


@pytest.mark.parametrize("shape", SHAPES[:3])
def test_solovev_safety_factor_axis_matches_a_traced_field_line(shape):
    psi0, r0, b0 = 0.1, 1.0, 1.0
    ax = solovev_axis(*shape)
    assert 1.0 < ax < 1.0 + shape[0]  # shifted outward
    s = solovev(ax, 0.0, r0, *shape, psi0, b0)
    assert abs(s.b_r) < 1e-12 and abs(s.b_z) < 1e-12

    def rhs(u):
        s = solovev(u[0], u[1], r0, *shape, psi0, b0)
        return np.array([u[0] * s.b_r / s.b_phi, u[0] * s.b_z / s.b_phi])

    u, phi, step = np.array([ax + 1e-3, 0.0]), 0.0, 0.02
    while True:  # half a poloidal turn: Z changes sign again
        k1 = rhs(u)
        k2 = rhs(u + step / 2 * k1)
        k3 = rhs(u + step / 2 * k2)
        k4 = rhs(u + step * k3)
        new = u + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if u[1] * new[1] < 0 and phi > 1:
            phi += u[1] / (u[1] - new[1]) * step
            break
        u, phi = new, phi + step
    assert 2 * phi / (2 * np.pi) == pytest.approx(solovev_safety_factor_axis(*shape, psi0, r0, b0), rel=1e-3)


def test_solovev_profiles_and_errors():
    shape = (0.3, 1.5, 0.2, 0.1)
    s = solovev(2.0, 0.0, 2.0, *shape, 0.3, 4.0)
    assert s.pressure > 0 and s.psi == pytest.approx(0.3 * s.psi_normalized)
    # outside the boundary the flux is positive: the boundary is where it changes sign
    assert solovev(2.0 * 1.45, 0.0, 2.0, *shape, 0.3, 4.0).psi_normalized > 0
    # pure vacuum field on the boundary: F = R₀ B₀
    r, z = solovev_boundary(*shape[:3], np.pi / 2, r0=2.0)
    assert solovev(r, z, 2.0, *shape, 0.3, 4.0).b_phi == pytest.approx(4.0 * 2.0 / r)
    for bad in [(1.2, 1.5, 0.2), (0.3, -1.0, 0.2), (0.3, 1.5, 1.0)]:
        with pytest.raises(ValueError):
            solovev_coefficients(*bad, 0.1)
    with pytest.raises(ValueError):
        solovev(1.0, 0.0, -1.0, *shape, 0.3, 4.0)


# ---------------------------------------------------------------------------------------------
# Force-free fields and the Taylor state
# ---------------------------------------------------------------------------------------------
def test_lundquist_is_force_free_and_divergence_free():
    alpha, b0 = 3.7, 0.4
    r = np.linspace(0.05, 2.0, 80)
    f = lundquist_field(r, alpha, b0)
    # (∇ × B)_θ = −dB_z/dr, (∇ × B)_z = (1/r) d(r B_θ)/dr: both equal α B
    curl_theta = -d(lambda t: lundquist_field(t, alpha, b0).b_z, r)
    curl_z = d(lambda t: t * lundquist_field(t, alpha, b0).b_theta, r) / r
    assert np.allclose(curl_theta, alpha * f.b_theta, atol=1e-8)
    assert np.allclose(curl_z, alpha * f.b_z, atol=1e-8)
    assert np.allclose(MU0 * f.current_density_theta, curl_theta, atol=1e-8)
    assert np.allclose(MU0 * f.current_density_z, curl_z, atol=1e-8)
    # J × B = 0: J ∥ B, and the force on the radial direction vanishes with p = 0
    force_r = f.current_density_theta * f.b_z - f.current_density_z * f.b_theta
    assert np.abs(force_r).max() < 1e-12 * np.abs(f.current_density_z).max()
    assert np.all(f.pressure == 0)


def test_taylor_state_parameters():
    s = taylor_state(2.404825557695773)
    assert s.pinch == pytest.approx(1.2024128, abs=1e-6)  # Θ at reversal (Taylor 1974)
    assert abs(s.reversal) < 1e-12
    # Θ and F by quadrature of the field
    alpha_a = 3.1
    r = np.linspace(0, 1, 200001)
    f = lundquist_field(r, alpha_a)
    mean_bz = np.trapezoid(f.b_z * 2 * r, r)
    t = taylor_state(alpha_a)
    assert t.pinch == pytest.approx(f.b_theta[-1] / mean_bz, rel=1e-9)
    assert t.reversal == pytest.approx(f.b_z[-1] / mean_bz, rel=1e-9)
    assert taylor_state(3.0).reversal < 0 < taylor_state(2.0).reversal
