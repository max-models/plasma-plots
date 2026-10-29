"""Exact MHD equilibria to verify MHD codes: the Harris current sheet, the Bennett pinch, the
Solov'ev tokamak equilibrium (Cerfon–Freidberg construction), and linear force-free (Lundquist)
fields with the Taylor state.

Every equilibrium satisfies the force balance ∇p = J × B with μ₀J = ∇ × B and ∇ · B = 0. The
tests check this by finite differences, which is the check to run on a code's own initial state
too. Profiles take the coordinates first, ``f(z, ...)``, so that a field of the returned result
(or a lambda around the function) can be passed as the ``reference=`` of plasma-plots' profile
plots. Everything is plain numpy and vectorized.

Units
-----
SI units (T, Pa, A/m², m) with temperatures in eV and densities in m⁻³, as in
:mod:`~plasma_plots.theory.parameters`. The Solov'ev functions take the flux scale and the
toroidal field in SI units; its shape parameters are dimensionless. The pressure is p = n k_B
(T_e + T_i) for a quasi-neutral plasma with n_e = n_i = n.

Harris sheet
------------
B = B₀ tanh(z/L) e_x. Pressure balance p + B²/2μ₀ = const gives p = (B₀²/2μ₀) sech²(z/L) + p_b,
and ∇ × B gives μ₀ J_y = (B₀/L) sech²(z/L). With A_y = −B₀ L ln cosh(z/L), B = ∇ × A.

Bennett pinch
-------------
With n = n₀/(1 + r²/a²)², the equation dp/dr = −J_z B_θ is solved by the enclosed current
I(r) = I r²/(r² + a²), so B_θ = μ₀ I r / (2π (r² + a²)), and the Bennett relation
μ₀ I² = 8π N k_B (T_e + T_i) holds for the line density N = π n₀ a².

Solov'ev equilibrium
--------------------
With x = R/R₀, y = Z/R₀, ψ̂ = Ψ/Ψ₀ (Ψ the poloidal flux per radian, B_R = −∂_Z Ψ/R, B_Z =
∂_R Ψ/R, B_φ = F/R) and the Solov'ev profiles μ₀ R₀⁴ p'/Ψ₀² = −(1 − A), R₀² F F'/Ψ₀² = −A, the
Grad–Shafranov equation Δ*ψ̂ = ψ̂_xx − ψ̂_x/x + ψ̂_yy = A + (1 − A) x² is linear. Its solution is
ψ̂ = A y²/2 + (1 − A) x⁴/8 + Σᵢ cᵢ ψᵢ with the homogeneous solutions Δ*ψᵢ = 0 of Cerfon and
Freidberg. The seven constants cᵢ make the up-down symmetric surface ψ̂ = 0 a "D" with inverse
aspect ratio ε, elongation κ and triangularity δ, by the conditions at the outer and inner
midplane points and the top point ``(1 − δε, κε)``: ψ̂ = 0 at all of them, ψ̂_x = 0 at the top and
the curvatures ψ̂_yy = −N ψ̂_x (outer, inner) and ψ̂_xx = −N₃ ψ̂_y (top) of the boundary
x = 1 + ε cos(τ + α sin τ), y = κ ε sin τ, with α = arcsin δ. The flux is negative inside, the
pressure p = (1 − A) Ψ₀² (−ψ̂)/(μ₀ R₀⁴) is positive and F² = F₀² + 2A (Ψ₀/R₀)² (−ψ̂), where F₀ is
the (vacuum) value at the boundary.

Force-free fields
-----------------
The linear force-free field ∇ × B = α B in a cylinder is B_θ = B₀ J₁(α r), B_z = B₀ J₀(α r)
(Lundquist). For a cylinder of radius a, the Taylor state, which minimises the magnetic energy
at fixed helicity, is this field; its pinch parameter is Θ = B_θ(a)/⟨B_z⟩ = α a/2 and its
reversal parameter F = B_z(a)/⟨B_z⟩ = α a J₀(α a) / (2 J₁(α a)) vanishes at α a = 2.405, the first
zero of J₀ (Θ = 1.202).

References
----------
H. Harris, "On a plane current sheet in a plasma", Nuovo Cimento 23, 115 (1962).

W. H. Bennett, "Magnetically self-focussing streams", Phys. Rev. 45, 890 (1934).

A. J. Cerfon and J. P. Freidberg, ""One size fits all" analytic solutions to the Grad–Shafranov
equation", Phys. Plasmas 17, 032502 (2010). The paper itself was not available to read; the
method (profiles, boundary conditions) is as I recall it, and the homogeneous solutions ψ₁ … ψ₇
are from their restatement in arXiv:2609.15074, table 1. Both are verified here: Δ*ψᵢ = 0,
the boundary shape, and the force balance are tested numerically.

J. P. Freidberg, Ideal MHD (Cambridge, 2014), chapters 6 and 9.

J. B. Taylor, "Relaxation of toroidal plasma and generation of reverse magnetic fields", Phys.
Rev. Lett. 33, 1139 (1974).

Examples
--------
The Harris-sheet peak density needed to balance a 10 mT field at T_e + T_i = 10 eV, and the
Bennett current of a 1e20 m⁻³ pinch of radius 1 mm at 100 eV (T_e + T_i):

>>> print(f"{harris_sheet(0.0, 0.01, 1.0, 5.0, 5.0).density:.4e}")
2.4834e+19
>>> print(f"{bennett_pinch(0.0, 1e20, 1e-3, 50.0, 50.0).total_current:.1f} A")
317.3 A
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plasma_plots.theory.parameters import elementary_charge, vacuum_permeability
from plasma_plots.theory.special import bessel_j

__all__ = [
    "BennettPinch",
    "HarrisSheet",
    "LundquistField",
    "SolovevFields",
    "TaylorState",
    "bennett_pinch",
    "harris_drift_speed",
    "harris_sheet",
    "lundquist_field",
    "solovev",
    "solovev_axis",
    "solovev_boundary",
    "solovev_coefficients",
    "solovev_safety_factor_axis",
    "taylor_state",
]

_MU0 = vacuum_permeability
_E = elementary_charge


def _out(a):
    """A numpy array, or a numpy scalar for 0-d input."""
    return np.asarray(a, dtype=float)[()]


# ---------------------------------------------------------------------------------------------
# Harris current sheet
# ---------------------------------------------------------------------------------------------
@dataclass
class HarrisSheet:
    """The Harris current sheet B = B₀ tanh(z/L) e_x.

    Attributes
    ----------
    b_x : float or numpy.ndarray
        The field B_x = B₀ tanh(z/L), in T.
    density : float or numpy.ndarray
        The density n = n₀ sech²(z/L) + n_b, in m⁻³.
    pressure : float or numpy.ndarray
        The pressure p = n k_B (T_e + T_i), in Pa.
    current_density : float or numpy.ndarray
        The current density J_y = (B₀/μ₀L) sech²(z/L), in A/m².
    vector_potential : float or numpy.ndarray
        The vector potential A_y = −B₀ L ln cosh(z/L), in T m.
    peak_density : float
        The sheet density n₀ = B₀²/(2μ₀ e (T_e + T_i)), without the background, in m⁻³.
    """

    b_x: object
    density: object
    pressure: object
    current_density: object
    vector_potential: object
    peak_density: float


def _log_cosh(u):
    """ln cosh u, without overflow."""
    u = np.abs(u)
    return u + np.log1p(np.exp(-2 * u)) - np.log(2)


def harris_sheet(
    z, b0, thickness, temperature_electron, temperature_ion, background_density=0.0
):
    """Compute the Harris current sheet: a field reversal in pressure balance.

    The field B_x = B₀ tanh(z/L) reverses across the sheet, the density n = n₀ sech²(z/L) + n_b
    peaks where it vanishes, and p + B²/2μ₀ is uniform when n₀ k_B (T_e + T_i) = B₀²/2μ₀ (the
    uniform background adds a uniform pressure). The current is μ₀ J_y = ∂_z B_x and the vector
    potential A_y = −B₀ L ln cosh(z/L) has B_x = −∂_z A_y.

    Parameters
    ----------
    z : float or array_like
        The coordinate across the sheet, in m.
    b0 : float
        The asymptotic field B₀, in T.
    thickness : float
        The half thickness L, in m.
    temperature_electron : float
        The electron temperature, in eV.
    temperature_ion : float
        The ion temperature, in eV.
    background_density : float, optional
        The uniform background density n_b, in m⁻³. Default: ``0.0``.

    Returns
    -------
    HarrisSheet
        The field, density, pressure, current density and vector potential.

    Raises
    ------
    ValueError
        If ``thickness`` or the temperatures are not positive, or the background density is
        negative.

    See Also
    --------
    harris_drift_speed : The drifts that carry the current.

    References
    ----------
    H. Harris, "On a plane current sheet in a plasma", Nuovo Cimento 23, 115 (1962).

    Examples
    --------
    The total pressure p + B²/2μ₀ is the same everywhere:

    >>> import numpy as np
    >>> s = harris_sheet(np.array([0.0, 0.5, 3.0]), 0.01, 1.0, 5.0, 5.0, 1e17)
    >>> total = s.pressure + s.b_x**2 / (2 * 1.25663706212e-06)
    >>> bool(np.ptp(total) < 1e-9 * total[0])
    True
    """
    if thickness <= 0 or temperature_electron <= 0 or temperature_ion <= 0:
        raise ValueError("thickness and the temperatures must be positive")
    if background_density < 0:
        raise ValueError(
            f"background_density must be non-negative; got {background_density!r}"
        )
    z = np.asarray(z, dtype=float)
    u = z / thickness
    n0 = b0**2 / (2 * _MU0 * _E * (temperature_electron + temperature_ion))
    sech2 = 4 * np.exp(-2 * np.abs(u)) / (1 + np.exp(-2 * np.abs(u))) ** 2
    density = n0 * sech2 + background_density
    return HarrisSheet(
        b_x=_out(b0 * np.tanh(u)),
        density=_out(density),
        pressure=_out(density * _E * (temperature_electron + temperature_ion)),
        current_density=_out(b0 / (_MU0 * thickness) * sech2),
        vector_potential=_out(-b0 * thickness * _log_cosh(u)),
        peak_density=float(n0),
    )


def harris_drift_speed(temperature, b0, thickness):
    """Compute the drift speed v = 2 T/(e B₀ L) with which a species carries the Harris current.

    With the species drifting along y at u_s, the current J_y = n₀ e (u_i + |u_e|) sech² equals the field's B₀ sech²/(μ₀ L) when
    u_s = 2 T_s/(e B₀ L) (T_s in eV; the Harris distribution), given n₀ (T_e + T_i) = B₀²/2μ₀.

    Parameters
    ----------
    temperature : float or array_like
        The temperature of the species, in eV.
    b0 : float
        The asymptotic field B₀, in T.
    thickness : float
        The half thickness L, in m.

    Returns
    -------
    float or numpy.ndarray
        The drift speed in m/s.

    References
    ----------
    H. Harris, "On a plane current sheet in a plasma", Nuovo Cimento 23, 115 (1962).

    Examples
    --------
    >>> print(f"{harris_drift_speed(5.0, 0.01, 1.0):.4e} m/s")
    1.0000e+03 m/s
    """
    return _out(2 * np.asarray(temperature, dtype=float) / (b0 * thickness))


# ---------------------------------------------------------------------------------------------
# Bennett pinch
# ---------------------------------------------------------------------------------------------
@dataclass
class BennettPinch:
    """The Bennett pinch.

    Attributes
    ----------
    density : float or numpy.ndarray
        The density n = n₀/(1 + r²/a²)², in m⁻³.
    pressure : float or numpy.ndarray
        The pressure n k_B (T_e + T_i), in Pa.
    b_theta : float or numpy.ndarray
        The azimuthal field, in T.
    current_density : float or numpy.ndarray
        The axial current density J_z = μ₀⁻¹ r⁻¹ d(r B_θ)/dr, in A/m².
    enclosed_current : float or numpy.ndarray
        The current inside r, I r²/(r² + a²), in A.
    total_current : float
        The total current I from the Bennett relation, in A.
    line_density : float
        The number of ions per unit length N = π n₀ a², in m⁻¹.
    """

    density: object
    pressure: object
    b_theta: object
    current_density: object
    enclosed_current: object
    total_current: float
    line_density: float


def bennett_pinch(r, density0, radius, temperature_electron, temperature_ion):
    """Compute the Bennett pinch: the isothermal z-pinch with n = n₀/(1 + r²/a²)².

    The current is fixed by the Bennett relation μ₀ I² = 8π N k_B (T_e + T_i) with the line
    density N = π n₀ a²: then dp/dr = −J_z B_θ holds at every r, with B_θ = μ₀ I r/(2π (r² +
    a²)) and J_z = I a²/(π (r² + a²)²).

    Parameters
    ----------
    r : float or array_like
        The radius, in m.
    density0 : float
        The density n₀ on the axis, in m⁻³.
    radius : float
        The scale radius a, in m.
    temperature_electron : float
        The electron temperature, in eV.
    temperature_ion : float
        The ion temperature, in eV.

    Returns
    -------
    BennettPinch
        Density, pressure, B_θ, current density and the currents.

    Raises
    ------
    ValueError
        If ``density0``, ``radius`` or the temperatures are not positive.

    References
    ----------
    W. H. Bennett, "Magnetically self-focussing streams", Phys. Rev. 45, 890 (1934).

    J. P. Freidberg, Ideal MHD (Cambridge, 2014), section 6.2.

    Examples
    --------
    The current and the edge field of a 1e20 m⁻³, 1 mm pinch at T_e + T_i = 100 eV:

    >>> p = bennett_pinch(1e-3, 1e20, 1e-3, 50.0, 50.0)
    >>> print(f"{p.total_current:.1f} A, B_theta(a) = {p.b_theta:.4e} T")
    317.3 A, B_theta(a) = 3.1728e-02 T
    """
    if (
        density0 <= 0
        or radius <= 0
        or temperature_electron <= 0
        or temperature_ion <= 0
    ):
        raise ValueError("density0, radius and the temperatures must be positive")
    r = np.asarray(r, dtype=float)
    kt = _E * (temperature_electron + temperature_ion)
    line_density = np.pi * density0 * radius**2
    current = np.sqrt(8 * np.pi * line_density * kt / _MU0)
    s = r**2 + radius**2
    density = density0 / (1 + (r / radius) ** 2) ** 2
    return BennettPinch(
        density=_out(density),
        pressure=_out(density * kt),
        b_theta=_out(_MU0 * current * r / (2 * np.pi * s)),
        current_density=_out(current * radius**2 / (np.pi * s**2)),
        enclosed_current=_out(current * r**2 / s),
        total_current=float(current),
        line_density=float(line_density),
    )


# ---------------------------------------------------------------------------------------------
# Solov'ev equilibrium (Cerfon and Freidberg)
# ---------------------------------------------------------------------------------------------
# The homogeneous solutions ψ₁ … ψ₇ of Δ*ψ = 0 as terms (coefficient, power of x, power of y,
# power of ln x).
_BASIS = (
    ((1, 0, 0, 0),),
    ((1, 2, 0, 0),),
    ((1, 0, 2, 0), (-1, 2, 0, 1)),
    ((1, 4, 0, 0), (-4, 2, 2, 0)),
    ((2, 0, 4, 0), (-9, 2, 2, 0), (3, 4, 0, 1), (-12, 2, 2, 1)),
    ((1, 6, 0, 0), (-12, 4, 2, 0), (8, 2, 4, 0)),
    (
        (8, 0, 6, 0),
        (-140, 2, 4, 0),
        (75, 4, 2, 0),
        (-15, 6, 0, 1),
        (180, 4, 2, 1),
        (-120, 2, 4, 1),
    ),
)


def _eval_terms(terms, x, y, dx=0, dy=0):
    """Evaluate Σ c xᵃ yᵇ lnᵏ x differentiated dx times in x and dy times in y."""
    total = np.zeros(np.broadcast(x, y).shape)
    logx = np.log(x)
    for c, a, b, k in terms:
        # y derivatives
        for _ in range(dy):
            c, b = c * b, b - 1
        if b < 0 or c == 0:
            continue
        # x derivatives of xᵃ lnᵏ x: a x^(a−1) lnᵏ + k x^(a−1) ln^(k−1)
        pieces = [(c, a, k)]
        for _ in range(dx):
            new = []
            for cc, aa, kk in pieces:
                if aa != 0:
                    new.append((cc * aa, aa - 1, kk))
                if kk > 0:
                    new.append((cc * kk, aa - 1, kk - 1))
            pieces = new
        for cc, aa, kk in pieces:
            total = total + cc * x**aa * y**b * logx**kk
    return total


def _basis(x, y, dx=0, dy=0):
    return np.array([_eval_terms(t, x, y, dx, dy) for t in _BASIS])


def _particular(x, y, a, dx=0, dy=0):
    """ψ_p = A y²/2 + (1 − A) x⁴/8 and its derivatives."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    if (dx, dy) == (0, 0):
        return a / 2 * y**2 + (1 - a) / 8 * x**4
    if (dx, dy) == (1, 0):
        return (1 - a) / 2 * x**3
    if (dx, dy) == (0, 1):
        return a * y
    if (dx, dy) == (2, 0):
        return 1.5 * (1 - a) * x**2
    if (dx, dy) == (0, 2):
        return a * np.ones_like(x)
    return np.zeros_like(x)  # (1, 1) and higher


def solovev_coefficients(epsilon, kappa, delta, a_param):
    """Compute the constants cᵢ that give a Solov'ev equilibrium a "D"-shaped boundary.

    Solves the 7 × 7 linear system of Cerfon and Freidberg for the up-down symmetric case: ψ̂
    vanishes at the outer midplane point (1 + ε, 0), the inner one (1 − ε, 0) and the top point
    (1 − δε, κε), ψ̂_x vanishes at the top point, and the curvatures of the surface ψ̂ = 0 at
    the three points match those of x = 1 + ε cos(τ + α sin τ), y = κε sin τ with α = arcsin δ
    (ψ̂_yy = −N₁ ψ̂_x at the outer point, ψ̂_yy = −N₂ ψ̂_x at the inner one, ψ̂_xx = −N₃ ψ̂_y
    at the top, with N₁ = −(1 + α)²/(εκ²), N₂ = (1 − α)²/(εκ²), N₃ = −κ/(ε cos²α) and
    α = arcsin δ).

    Parameters
    ----------
    epsilon : float
        The inverse aspect ratio ε = a/R₀, in (0, 1).
    kappa : float
        The elongation κ > 0.
    delta : float
        The triangularity δ, in (−1, 1).
    a_param : float
        The constant A of the Solov'ev profiles: A = 0 is pressure-driven, A = 1 force-free.

    Returns
    -------
    numpy.ndarray
        The seven coefficients c₁ … c₇ of ψ̂ = A y²/2 + (1 − A) x⁴/8 + Σ cᵢ ψᵢ.

    Raises
    ------
    ValueError
        If ``epsilon`` is not in (0, 1), ``kappa`` ≤ 0 or ``delta`` is not in (−1, 1).

    See Also
    --------
    solovev : The fields.

    References
    ----------
    A. J. Cerfon and J. P. Freidberg, Phys. Plasmas 17, 032502 (2010).

    Examples
    --------
    The seven constants of an ITER-like shape:

    >>> c = solovev_coefficients(0.32, 1.7, 0.33, -0.155)
    >>> c.shape
    (7,)
    """
    if not 0 < epsilon < 1 or kappa <= 0 or not -1 < delta < 1:
        raise ValueError("need 0 < epsilon < 1, kappa > 0 and -1 < delta < 1")
    al = np.arcsin(delta)
    n1 = -((1 + al) ** 2) / (epsilon * kappa**2)
    n2 = (1 - al) ** 2 / (epsilon * kappa**2)
    n3 = -kappa / (epsilon * np.cos(al) ** 2)
    xo, xi, xt, yt = 1 + epsilon, 1 - epsilon, 1 - delta * epsilon, kappa * epsilon
    rows, rhs = [], []

    def add(row, particular):
        rows.append(row)
        rhs.append(-particular)

    add(_basis(xo, 0.0), _particular(xo, 0.0, a_param))
    add(_basis(xi, 0.0), _particular(xi, 0.0, a_param))
    add(_basis(xt, yt), _particular(xt, yt, a_param))
    add(_basis(xt, yt, 1, 0), _particular(xt, yt, a_param, 1, 0))
    add(
        _basis(xo, 0.0, 0, 2) + n1 * _basis(xo, 0.0, 1, 0),
        _particular(xo, 0.0, a_param, 0, 2) + n1 * _particular(xo, 0.0, a_param, 1, 0),
    )
    add(
        _basis(xi, 0.0, 0, 2) + n2 * _basis(xi, 0.0, 1, 0),
        _particular(xi, 0.0, a_param, 0, 2) + n2 * _particular(xi, 0.0, a_param, 1, 0),
    )
    add(
        _basis(xt, yt, 2, 0) + n3 * _basis(xt, yt, 0, 1),
        _particular(xt, yt, a_param, 2, 0) + n3 * _particular(xt, yt, a_param, 0, 1),
    )
    return np.linalg.solve(np.array(rows, dtype=float), np.array(rhs, dtype=float))


def solovev_boundary(epsilon, kappa, delta, tau=None, r0=1.0):
    """Compute the "D"-shaped boundary x = 1 + ε cos(τ + α sin τ), y = κ ε sin τ.

    Parameters
    ----------
    epsilon : float
        The inverse aspect ratio ε.
    kappa : float
        The elongation κ.
    delta : float
        The triangularity δ, with α = arcsin δ.
    tau : array_like, optional
        The angles τ. Default: 129 points on [0, 2π].
    r0 : float, optional
        The major radius R₀ that scales R = R₀ x and Z = R₀ y. Default: ``1.0``.

    Returns
    -------
    tuple of numpy.ndarray
        The boundary points (R, Z).

    References
    ----------
    A. J. Cerfon and J. P. Freidberg, Phys. Plasmas 17, 032502 (2010).

    Examples
    --------
    The outer, top and inner points for ε = 0.3, κ = 1.5, δ = 0.2:

    >>> r, z = solovev_boundary(0.3, 1.5, 0.2, [0.0, np.pi / 2, np.pi])
    >>> r.round(4), (z + 0.0).round(4)
    (array([1.3 , 0.94, 0.7 ]), array([0.  , 0.45, 0.  ]))
    """
    tau = (
        np.linspace(0.0, 2 * np.pi, 129)
        if tau is None
        else np.asarray(tau, dtype=float)
    )
    al = np.arcsin(delta)
    return r0 * (
        1 + epsilon * np.cos(tau + al * np.sin(tau))
    ), r0 * kappa * epsilon * np.sin(tau)


@dataclass
class SolovevFields:
    """The Solov'ev equilibrium on a grid of (R, Z).

    Attributes
    ----------
    psi_normalized : float or numpy.ndarray
        The normalized flux ψ̂ = Ψ/Ψ₀: zero on the boundary, negative inside.
    psi : float or numpy.ndarray
        The poloidal flux per radian Ψ, in Wb.
    b_r : float or numpy.ndarray
        The radial field −∂_Z Ψ/R, in T.
    b_z : float or numpy.ndarray
        The vertical field ∂_R Ψ/R, in T.
    b_phi : float or numpy.ndarray
        The toroidal field F/R, in T.
    pressure : float or numpy.ndarray
        The pressure, in Pa.
    current_density_phi : float or numpy.ndarray
        The toroidal current density −(Ψ₀/μ₀R₀³)(A/x + (1 − A) x) = −Δ*Ψ/(μ₀R), in A/m².
    """

    psi_normalized: object
    psi: object
    b_r: object
    b_z: object
    b_phi: object
    pressure: object
    current_density_phi: object


def solovev(r, z, r0, epsilon, kappa, delta, a_param, psi0, b0):
    """Compute the Solov'ev equilibrium of Cerfon and Freidberg: ψ, B and p of a shaped tokamak.

    The Grad–Shafranov solution ψ̂ = A y²/2 + (1 − A) x⁴/8 + Σ cᵢ ψᵢ, x = R/R₀, y = Z/R₀, with
    the constants from :func:`solovev_coefficients`; B_R = −∂_Z Ψ/R, B_Z = ∂_R Ψ/R,
    B_φ = F/R with F² = F₀² + 2A (Ψ₀/R₀)² (−ψ̂) and F₀ = R₀ B₀, and the pressure
    p = (1 − A) Ψ₀² (−ψ̂)/(μ₀ R₀⁴). The pressure vanishes on the boundary and the points outside
    it (ψ̂ > 0) are not part of the plasma: there ψ̂ and the fields continue analytically but p
    is negative, so mask them with ``psi_normalized < 0``.

    Parameters
    ----------
    r : float or array_like
        The major radius R, in m.
    z : float or array_like
        The height Z, in m.
    r0 : float
        The reference major radius R₀, in m.
    epsilon : float
        The inverse aspect ratio ε.
    kappa : float
        The elongation κ.
    delta : float
        The triangularity δ.
    a_param : float
        The constant A of the profiles: A = 0 is pressure-driven, A = 1 force-free.
    psi0 : float
        The flux scale Ψ₀ in Wb (its sign sets the direction of the poloidal field).
    b0 : float
        The vacuum toroidal field at R₀, in T: F₀ = R₀ B₀ is F on the boundary.

    Returns
    -------
    SolovevFields
        ψ̂, Ψ, B_R, B_Z, B_φ, the pressure and the toroidal current density.

    Raises
    ------
    ValueError
        If the shape parameters are out of range (see :func:`solovev_coefficients`), or ``r0``
        ≤ 0.

    See Also
    --------
    solovev_axis : The magnetic axis.
    solovev_safety_factor_axis : q on the axis.

    References
    ----------
    A. J. Cerfon and J. P. Freidberg, Phys. Plasmas 17, 032502 (2010).

    L. S. Solov'ev, "The theory of hydromagnetic stability of toroidal plasma configurations",
    Sov. Phys. JETP 26, 400 (1968).

    Examples
    --------
    The flux vanishes on the boundary and is negative on the axis:

    >>> s = solovev(1.3, 0.0, 1.0, 0.3, 1.5, 0.2, 0.1, 0.1, 1.0)
    >>> abs(float(s.psi_normalized)) < 1e-12
    True
    >>> s = solovev(1.0, 0.0, 1.0, 0.3, 1.5, 0.2, 0.1, 0.1, 1.0)
    >>> float(s.psi_normalized) < 0 and float(s.pressure) > 0
    True
    """
    if r0 <= 0:
        raise ValueError(f"r0 must be positive; got {r0!r}")
    c = solovev_coefficients(epsilon, kappa, delta, a_param)
    x, y = np.broadcast_arrays(
        np.asarray(r, dtype=float) / r0, np.asarray(z, dtype=float) / r0
    )

    def derivative(dx, dy):
        return _particular(x, y, a_param, dx, dy) + np.tensordot(
            c, _basis(x, y, dx, dy), axes=1
        )

    psi_hat = derivative(0, 0)
    # d/dR = (1/R0) d/dx and Ψ = Ψ₀ ψ̂, so B_R = −Ψ₀ ψ̂_y/(R0² x) and B_Z = Ψ₀ ψ̂_x/(R0² x)
    scale = psi0 / r0**2
    f2 = (r0 * b0) ** 2 + 2 * a_param * (psi0 / r0) ** 2 * (-psi_hat)
    return SolovevFields(
        psi_normalized=_out(psi_hat),
        psi=_out(psi0 * psi_hat),
        b_r=_out(-scale * derivative(0, 1) / x),
        b_z=_out(scale * derivative(1, 0) / x),
        b_phi=_out(np.sqrt(f2) / (r0 * x)),
        pressure=_out((1 - a_param) * psi0**2 * (-psi_hat) / (_MU0 * r0**4)),
        current_density_phi=_out(
            -psi0 / (_MU0 * r0**3) * (a_param / x + (1 - a_param) * x)
        ),
    )


def solovev_axis(epsilon, kappa, delta, a_param, r0=1.0):
    """Find the magnetic axis (R_ax, 0) of a Solov'ev equilibrium, where ∇ψ = 0.

    Newton's method on ψ̂_x(x, 0) = 0, starting at the geometric centre x = 1 (by up-down
    symmetry the axis is on the midplane).

    Parameters
    ----------
    epsilon : float
        The inverse aspect ratio ε.
    kappa : float
        The elongation κ.
    delta : float
        The triangularity δ.
    a_param : float
        The constant A of the profiles.
    r0 : float, optional
        The reference major radius R₀. Default: ``1.0``.

    Returns
    -------
    float
        The major radius of the axis, in the units of ``r0``.

    Raises
    ------
    RuntimeError
        If Newton's method does not converge.

    References
    ----------
    A. J. Cerfon and J. P. Freidberg, Phys. Plasmas 17, 032502 (2010).

    Examples
    --------
    The axis is shifted outward of the geometric centre 1:

    >>> print(f"{solovev_axis(0.32, 1.7, 0.33, -0.155):.4f}")
    1.0513
    """
    c = solovev_coefficients(epsilon, kappa, delta, a_param)
    x = 1.0
    for _ in range(50):
        g = _particular(x, 0.0, a_param, 1, 0) + c @ _basis(x, 0.0, 1, 0)
        h = _particular(x, 0.0, a_param, 2, 0) + c @ _basis(x, 0.0, 2, 0)
        step = g / h
        x = float(x - step)
        if abs(step) < 1e-14:
            return x * r0
    raise RuntimeError("the magnetic axis search did not converge")


def solovev_safety_factor_axis(epsilon, kappa, delta, a_param, psi0, r0, b0):
    """Compute the safety factor on the magnetic axis, q₀ = F/(R √(Ψ_RR Ψ_ZZ)).

    Near the axis the flux surfaces are ellipses of area dA = 2π dΨ/√(Ψ_RR Ψ_ZZ), and q = (1/2π)
    ∮ F dl/(R² B_p) = (F/2πR) dA/dΨ, evaluated at R = R_ax.

    Parameters
    ----------
    epsilon : float
        The inverse aspect ratio ε.
    kappa : float
        The elongation κ.
    delta : float
        The triangularity δ.
    a_param : float
        The constant A of the profiles.
    psi0 : float
        The flux scale Ψ₀, in Wb (only its magnitude matters).
    r0 : float
        The reference major radius R₀, in m.
    b0 : float
        The vacuum toroidal field at R₀, in T.

    Returns
    -------
    float
        q₀ (positive).

    References
    ----------
    J. P. Freidberg, Ideal MHD (Cambridge, 2014), section 6.7.

    Examples
    --------
    >>> print(
    ...     f"{solovev_safety_factor_axis(0.32, 1.7, 0.33, -0.155, 0.1, 1.0, 1.0):.4f}"
    ... )
    18.8074
    """
    c = solovev_coefficients(epsilon, kappa, delta, a_param)
    x = solovev_axis(epsilon, kappa, delta, a_param)
    pxx = _particular(x, 0.0, a_param, 2, 0) + c @ _basis(x, 0.0, 2, 0)
    pyy = _particular(x, 0.0, a_param, 0, 2) + c @ _basis(x, 0.0, 0, 2)
    psi_ax = _particular(x, 0.0, a_param) + c @ _basis(x, 0.0)
    f_ax = np.sqrt((r0 * b0) ** 2 + 2 * a_param * (psi0 / r0) ** 2 * (-psi_ax))
    # Ψ_RR = Ψ₀ ψ̂_xx / R₀², so √(Ψ_RR Ψ_ZZ) = |Ψ₀| √(ψ̂_xx ψ̂_yy)/R₀²
    return float(f_ax / (x * r0 * abs(psi0) / r0**2 * np.sqrt(pxx * pyy)))


# ---------------------------------------------------------------------------------------------
# Force-free fields and the Taylor state
# ---------------------------------------------------------------------------------------------
@dataclass
class LundquistField:
    """The linear force-free field ∇ × B = α B in a cylinder.

    Attributes
    ----------
    b_theta : float or numpy.ndarray
        The azimuthal field B₀ J₁(α r), in T.
    b_z : float or numpy.ndarray
        The axial field B₀ J₀(α r), in T.
    current_density_theta : float or numpy.ndarray
        The azimuthal current density α B_θ/μ₀, in A/m².
    current_density_z : float or numpy.ndarray
        The axial current density α B_z/μ₀, in A/m².
    pressure : float or numpy.ndarray
        The pressure, zero (the field is force free).
    """

    b_theta: object
    b_z: object
    current_density_theta: object
    current_density_z: object
    pressure: object


def lundquist_field(r, alpha, b0=1.0):
    """Compute the Lundquist field B = B₀ (J₁(α r) e_θ + J₀(α r) e_z): force free with J = α B/μ₀.

    Parameters
    ----------
    r : float or array_like
        The radius, in m.
    alpha : float
        The force-free constant α in ∇ × B = α B, in 1/m.
    b0 : float, optional
        The axial field on the axis, in T. Default: ``1.0``.

    Returns
    -------
    LundquistField
        B_θ, B_z, the current density and the (zero) pressure.

    References
    ----------
    S. Lundquist, "Magneto-hydrostatic fields", Ark. Fys. 2, 361 (1950).

    Examples
    --------
    The field vanishes in B_z at the first zero of J₀, r = 2.405/α:

    >>> f = lundquist_field(2.404825557695773, 1.0)
    >>> abs(float(f.b_z)) < 1e-12, round(float(f.b_theta), 6)
    (True, 0.519147)
    """
    r = np.asarray(r, dtype=float)
    bt = b0 * bessel_j(1, alpha * r)
    bz = b0 * bessel_j(0, alpha * r)
    return LundquistField(
        b_theta=_out(bt),
        b_z=_out(bz),
        current_density_theta=_out(alpha * bt / _MU0),
        current_density_z=_out(alpha * bz / _MU0),
        pressure=_out(np.zeros_like(r)),
    )


@dataclass
class TaylorState:
    """The pinch and reversal parameters of the Taylor state.

    Attributes
    ----------
    pinch : float or numpy.ndarray
        Θ = B_θ(a)/⟨B_z⟩ = α a/2.
    reversal : float or numpy.ndarray
        F = B_z(a)/⟨B_z⟩ = α a J₀(α a)/(2 J₁(α a)).
    """

    pinch: object
    reversal: object


def taylor_state(alpha_a):
    """Compute the pinch parameter Θ and reversal parameter F of the Taylor state of α a.

    Taylor's relaxed state in a cylinder of radius a is the Lundquist field with ⟨B_z⟩ = 2B₀
    J₁(αa)/(αa), so Θ = αa/2 and F = αa J₀(αa)/(2 J₁(αa)). The field reverses at the wall (F < 0)
    for αa > 2.405, the first zero of J₀, i.e. Θ > 1.202.

    Parameters
    ----------
    alpha_a : float or array_like
        The product of the force-free constant and the wall radius, α a (not a zero of J₁).

    Returns
    -------
    TaylorState
        The pinch and reversal parameters.

    References
    ----------
    J. B. Taylor, Phys. Rev. Lett. 33, 1139 (1974).

    Examples
    --------
    Reversal sets in at Θ = 1.2024:

    >>> s = taylor_state(2.404825557695773)
    >>> round(float(s.pinch), 4), abs(float(s.reversal)) < 1e-12
    (1.2024, True)
    """
    x = np.asarray(alpha_a, dtype=float)
    return TaylorState(
        pinch=_out(x / 2), reversal=_out(x * bessel_j(0, x) / (2 * bessel_j(1, x)))
    )
