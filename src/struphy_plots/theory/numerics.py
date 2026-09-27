"""Accuracy of the numerics: amplification and phase errors of time integrators, and the numerical
dispersion of spline Galerkin discretizations.

Use these to judge a simulation result: is the energy drift or the frequency error of a run what
the scheme is expected to produce? All functions are plain numpy and
vectorized; frequencies follow the package convention ``exp(i(kx − ωt))`` (a positive imaginary
part is numerical growth, a negative one numerical damping).

The time integrators are characterized on the oscillator y' = iωy, the test equation of every
linear wave: one step of size dt multiplies y by the amplification factor G(ω dt), which for the
exact flow is exp(iω dt). Names of the methods (``method=``):

* ``"explicit_euler"`` (``"forward_euler"``): G = 1 + z, with z = iω dt.
* ``"implicit_euler"`` (``"backward_euler"``): G = 1/(1 − z).
* ``"implicit_midpoint"`` (``"crank_nicolson"``, ``"trapezoidal"``, ``"discrete_gradient"``):
  G = (1 + z/2)/(1 − z/2). For a linear (quadratic-energy) problem the discrete-gradient methods
  (average vector field, Gonzalez) coincide with it.
* ``"rk2"`` (``"heun"``), ``"rk3"`` (``"ssprk3"``), ``"rk4"``: explicit Runge–Kutta methods with s
  = order stages, whose G = Σ_{n ≤ s} zⁿ/n! does not depend on the Butcher tableau.
* ``"leapfrog"`` (``"stormer_verlet"``, ``"verlet"``): the staggered leapfrog / Störmer–Verlet
  scheme for x'' = −ω²x (the oscillator as the pair x' = v, v' = −ω²x, also the Yee scheme),
  whose one-step map has the eigenvalues G and 1/G with
  G = 1 − (ω dt)²/2 + iω dt √(1 − (ω dt)²/4), stable for |ω dt| ≤ 2.
* ``"two_step_leapfrog"``: the explicit midpoint rule y_{n+1} = y_{n−1} + 2z y_n on the complex
  first-order equation, principal root G = z + √(1 + z²), stable for |ω dt| ≤ 1 (its second root
  −1/G is the computational mode).
"""

from __future__ import annotations

from math import comb, factorial

import numpy as np

# stability polynomials / rational functions in z = iω dt: (numerator, denominator) coefficients,
# lowest power first; G(0) = 1
_RATIONAL = {
    "explicit_euler": ((1.0, 1.0), (1.0,)),
    "implicit_euler": ((1.0,), (1.0, -1.0)),
    "implicit_midpoint": ((1.0, 0.5), (1.0, -0.5)),
    "rk2": ((1.0, 1.0, 1 / 2), (1.0,)),
    "rk3": ((1.0, 1.0, 1 / 2, 1 / 6), (1.0,)),
    "rk4": ((1.0, 1.0, 1 / 2, 1 / 6, 1 / 24), (1.0,)),
}
_ALIASES = {
    "forward_euler": "explicit_euler",
    "backward_euler": "implicit_euler",
    "crank_nicolson": "implicit_midpoint",
    "trapezoidal": "implicit_midpoint",
    "discrete_gradient": "implicit_midpoint",
    "heun": "rk2",
    "ssprk3": "rk3",
    "stormer_verlet": "leapfrog",
    "verlet": "leapfrog",
}
_STABILITY_LIMITS = {
    "explicit_euler": 0.0,
    "implicit_euler": np.inf,
    "implicit_midpoint": np.inf,
    "rk2": 0.0,
    "rk3": np.sqrt(3.0),
    "rk4": 2 * np.sqrt(2.0),
    "leapfrog": 2.0,
    "two_step_leapfrog": 1.0,
}
_METHODS = sorted(set(_STABILITY_LIMITS) | set(_ALIASES))


def _method(method: str) -> str:
    name = str(method).lower().replace("-", "_").replace(" ", "_")
    name = _ALIASES.get(name, name)
    if name not in _STABILITY_LIMITS:
        raise ValueError(f"unknown method {method!r}; choose one of {', '.join(_METHODS)}")
    return name


def _factor_and_phase(theta, method: str):
    """G(iθ) and its continuous argument (unwrapped along θ from 0)."""
    name = _method(method)
    theta = np.asarray(theta, dtype=float)
    if name in _RATIONAL:
        numerator, denominator = _RATIONAL[name]
        z = 1j * theta
        g = np.polynomial.polynomial.polyval(z, numerator) / np.polynomial.polynomial.polyval(z, denominator)
        # G = Π(1 − z/r) / Π(1 − z/q) since G(0) = 1; each factor moves on a straight line from 1 that
        # misses 0, so its principal argument is continuous and the sum is the unwrapped phase
        phase = np.zeros_like(theta)
        for coefficients, sign in ((numerator, 1.0), (denominator, -1.0)):
            for root in np.polynomial.polynomial.polyroots(coefficients) if len(coefficients) > 1 else []:
                phase = phase + sign * np.angle(1 - z / root)
        return g, phase
    a = np.abs(theta)
    s = np.sign(theta)
    if name == "leapfrog":
        c = 1 - theta**2 / 2
        stable = a <= 2
        root = np.sqrt(np.abs(1 - theta**2 / 4))
        g = np.where(stable, c + 1j * theta * root, c - a * root + 0j)
        phase = np.where(stable, s * np.arccos(np.clip(c, -1, 1)), s * np.pi)
        return g, phase
    # two-step leapfrog
    stable = a <= 1
    root = np.sqrt(np.abs(1 - theta**2))
    g = np.where(stable, 1j * theta + root, 1j * s * (a + root))
    phase = np.where(stable, np.arcsin(np.clip(theta, -1, 1)), s * np.pi / 2)
    return g, phase


def _components(x):
    """A list of per-direction arrays from a scalar/array or a list/tuple of them."""
    if isinstance(x, (list, tuple)):
        return [np.asarray(v, dtype=float) for v in x]
    return [np.asarray(x, dtype=float)]


def _combine(rates, combine: str):
    if combine == "sum":
        return sum(rates)
    if combine == "max":
        return np.maximum.reduce(np.broadcast_arrays(*rates))
    if combine == "euclidean":
        return np.sqrt(sum(r**2 for r in rates))
    raise ValueError(f"combine must be 'sum', 'max' or 'euclidean', not {combine!r}")


def _rates(dx, speed):
    dxs = _components(dx)
    speeds = _components(speed)
    if len(speeds) == 1:
        speeds = speeds * len(dxs)
    if len(speeds) != len(dxs):
        raise ValueError(f"got {len(speeds)} speeds for {len(dxs)} grid spacings")
    return [np.abs(v) / h for v, h in zip(speeds, dxs)]


# ---------------------------------------------------------------------------------------------
# CFL
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Time integrators
# ---------------------------------------------------------------------------------------------
def amplification_factor(omega_dt, method: str):
    """Compute the amplification factor of a time integrator on the oscillator y' = iωy.

    One step multiplies y by G(ω dt); the exact flow has G = exp(iω dt). See the module
    documentation for the formula of each method. For ``"leapfrog"`` / ``"stormer_verlet"`` G is
    the principal eigenvalue of the one-step map of Störmer–Verlet on x'' = −ω²x (the one that
    tends to exp(iω dt)); beyond the stability limit |ω dt| > 2 it is the growing real root. For
    ``"two_step_leapfrog"`` it is the principal root of G² − 2iω dt G − 1 = 0.

    Parameters
    ----------
    omega_dt : float or array_like
        The (real) frequency times the time step, ω dt.
    method : str
        The integrator, e.g. ``"implicit_midpoint"``, ``"rk4"``, ``"leapfrog"``; see the module
        documentation for all names.

    Returns
    -------
    complex or numpy.ndarray
        G, complex.

    Raises
    ------
    ValueError
        If ``method`` is unknown.

    See Also
    --------
    amplitude_error, phase_error, numerical_frequency

    References
    ----------
    E. Hairer, C. Lubich and G. Wanner, "Geometric Numerical Integration", 2nd ed., Springer
    (2006), Sec. I.1 and I.3. E. Hairer and G. Wanner, "Solving Ordinary Differential Equations
    II", 2nd ed., Springer (1996), Sec. IV.2. D. R. Durran, "Numerical Methods for Fluid
    Dynamics", 2nd ed., Springer (2010), Ch. 2.

    Examples
    --------
    >>> g = amplification_factor(0.5, "implicit_midpoint")
    >>> print(round(abs(g), 12))
    1.0
    >>> print(np.round(amplification_factor([0.5, 1.0], "rk4"), 4))
    [0.8776+0.4792j 0.5417+0.8333j]
    """
    return _factor_and_phase(omega_dt, method)[0][()]


def amplitude_error(omega_dt, method: str):
    """Compute the relative amplitude error per step, |G| − 1, of a time integrator.

    Negative values are numerical damping, positive ones numerical growth, per step of size dt.
    The corresponding growth rate per unit time is ln|G|/dt, the imaginary part of
    :func:`numerical_frequency`. Examples: explicit Euler |G| = √(1 + (ω dt)²), RK2
    |G| − 1 ≈ (ω dt)⁴/8, RK3 ≈ −(ω dt)⁴/24, RK4 ≈ −(ω dt)⁶/144, implicit midpoint and (stable) leapfrog |G| = 1 exactly.

    Parameters
    ----------
    omega_dt : float or array_like
        The frequency times the time step, ω dt.
    method : str
        The integrator; see :func:`amplification_factor`.

    Returns
    -------
    float or numpy.ndarray
        |G| − 1.

    Raises
    ------
    ValueError
        If ``method`` is unknown.

    See Also
    --------
    amplification_factor, phase_error

    References
    ----------
    E. Hairer, C. Lubich and G. Wanner, "Geometric Numerical Integration", 2nd ed., Springer
    (2006), Sec. I.1.

    Examples
    --------
    >>> print(round(float(amplitude_error(0.1, "explicit_euler")), 6))
    0.004988
    >>> print(round(float(amplitude_error(0.1, "implicit_midpoint")), 12))
    0.0
    """
    name = _method(method)
    theta = np.asarray(omega_dt, dtype=float)
    g = _factor_and_phase(theta, name)[0]
    if name not in _RATIONAL:
        return (np.abs(g) - 1)[()]
    # |G|² − 1 = (N(z)N(−z) − D(z)D(−z)) / (D(z)D(−z)) at z = iθ, with the constant term cancelled
    # exactly, so that small errors (RK4: θ⁶/144) are not lost to rounding
    P = np.polynomial.Polynomial
    numerator, denominator = (P(c) for c in _RATIONAL[name])
    reflect = P([0.0, -1.0])
    difference = (numerator * numerator(reflect) - denominator * denominator(reflect)).coef.copy()
    difference[0] = 0.0
    z = 1j * theta
    squared_minus_one = (P(difference)(z) / np.abs(denominator(z)) ** 2).real
    return (squared_minus_one / (np.abs(g) + 1))[()]


def phase_error(omega_dt, method: str):
    """Compute the relative frequency error of the numerical oscillation of a time integrator.

    ε = arg G/(ω dt) − 1 = Re(ω_num)/ω − 1, with the argument of G unwrapped continuously from
    ω dt = 0. Negative values mean the numerical oscillation is too slow (a phase lag). Leading
    terms: explicit and implicit Euler −(ω dt)²/3, implicit midpoint −(ω dt)²/12, leapfrog
    +(ω dt)²/24, RK2 +(ω dt)²/6, RK3 +(ω dt)⁴/30, RK4 −(ω dt)⁴/120.

    Parameters
    ----------
    omega_dt : float or array_like
        The frequency times the time step, ω dt.
    method : str
        The integrator; see :func:`amplification_factor`.

    Returns
    -------
    float or numpy.ndarray
        The relative frequency error (0 at ω dt = 0).

    Raises
    ------
    ValueError
        If ``method`` is unknown.

    See Also
    --------
    numerical_frequency : The complex frequency the scheme produces.

    References
    ----------
    E. Hairer, C. Lubich and G. Wanner, "Geometric Numerical Integration", 2nd ed., Springer
    (2006), Sec. I.1. D. R. Durran, "Numerical Methods for Fluid Dynamics", 2nd ed., Springer
    (2010), Ch. 2.

    Examples
    --------
    >>> print(round(float(phase_error(0.1, "implicit_midpoint")), 7))
    -0.0008321
    >>> print(round(float(phase_error(0.1, "leapfrog")), 7))
    0.0004171
    """
    theta = np.asarray(omega_dt, dtype=float)
    phase = _factor_and_phase(theta, method)[1]
    small = np.abs(theta) < 1e-300
    return np.where(small, 0.0, phase / np.where(small, 1.0, theta) - 1)[()]


def stability_limit(method: str) -> float:
    """Return the largest stable |ω dt| of a time integrator on the imaginary axis.

    The scheme keeps |G| ≤ 1 on y' = iωy for |ω dt| up to this value: 0 for explicit Euler and
    RK2 (always weakly unstable, |G|² = 1 + (ω dt)² and 1 + (ω dt)⁴/4), √3 for RK3, 2√2 for RK4,
    2 for leapfrog / Störmer–Verlet, 1 for the two-step leapfrog, infinite for implicit Euler
    (damped) and implicit midpoint (exactly |G| = 1).

    Parameters
    ----------
    method : str
        The integrator; see :func:`amplification_factor`.

    Returns
    -------
    float
        The stability limit of ω dt (``inf`` for unconditionally stable methods).

    Raises
    ------
    ValueError
        If ``method`` is unknown.

    References
    ----------
    E. Hairer and G. Wanner, "Solving Ordinary Differential Equations II", 2nd ed., Springer
    (1996), Sec. IV.2. D. R. Durran, "Numerical Methods for Fluid Dynamics", 2nd ed., Springer
    (2010), Ch. 2.

    Examples
    --------
    >>> print(round(stability_limit("rk4"), 4))
    2.8284
    >>> stability_limit("crank_nicolson")
    inf
    """
    return float(_STABILITY_LIMITS[_method(method)])


def numerical_frequency(omega, dt, method: str):
    """Compute the complex frequency that a time integrator produces for a real frequency ω.

    The numerical solution of an oscillation at frequency ω behaves as exp(−iω_num t) at the
    time steps, with ω_num = (arg G + i ln|G|)/dt and G = G(ω dt) the amplification factor (the
    argument unwrapped continuously from 0). The real part is the frequency seen in the spectrum
    of a run, the imaginary part the numerical growth (> 0) or damping (< 0) rate. Beyond the
    stability limit of the explicit leapfrog schemes the real part locks to the Nyquist
    frequency π/dt (π/(2 dt) for the two-step leapfrog).

    Combine it with a spatial dispersion relation for the fully discrete frequency, e.g.
    ``numerical_frequency(spline_galerkin_dispersion(k, dx, 3), dt, "crank_nicolson")`` for a 1-D
    spline wave equation advanced with Crank–Nicolson.

    Parameters
    ----------
    omega : float or array_like
        The (real) frequency of the semi-discrete (time-continuous) problem.
    dt : float or array_like
        The time step.
    method : str
        The integrator; see :func:`amplification_factor`.

    Returns
    -------
    complex or numpy.ndarray
        ω_num, complex.

    Raises
    ------
    ValueError
        If ``method`` is unknown.

    See Also
    --------
    amplification_factor, phase_error

    References
    ----------
    E. Hairer, C. Lubich and G. Wanner, "Geometric Numerical Integration", 2nd ed., Springer
    (2006), Sec. I.1. C. K. Birdsall and A. B. Langdon, "Plasma Physics via Computer Simulation",
    IOP Publishing (1991), Ch. 4.

    Examples
    --------
    Crank–Nicolson is slow but undamped, RK2 fast and growing:

    >>> w = numerical_frequency(1.0, 0.5, "implicit_midpoint")
    >>> print(round(w.real, 5), abs(w.imag) < 1e-12)
    0.97991 True
    >>> w = numerical_frequency(1.0, 0.5, "rk2")
    >>> print(round(w.real, 5), round(w.imag, 5))
    1.03829 0.0155
    """
    omega = np.asarray(omega, dtype=float)
    dt = np.asarray(dt, dtype=float)
    g, phase = _factor_and_phase(omega * dt, method)
    with np.errstate(divide="ignore"):
        return ((phase + 1j * np.log(np.abs(g))) / dt)[()]


# ---------------------------------------------------------------------------------------------
# Spatial discretizations
# ---------------------------------------------------------------------------------------------
# central-difference stencils: first derivative Σ a_j sin(jθ) / h, second Σ b_j (2 − 2 cos jθ) / h²
_FD_FIRST = {2: (1.0,), 4: (8 / 6, -1 / 6), 6: (45 / 30, -9 / 30, 1 / 30)}
_FD_SECOND = {2: (1.0,), 4: (16 / 12, -1 / 12), 6: (270 / 180, -27 / 180, 2 / 180)}


def _cardinal_bspline(n: int, x: float) -> float:
    """The cardinal B-spline of degree n (support [0, n + 1]) at x."""
    return sum((-1) ** i * comb(n + 1, i) * max(x - i, 0.0) ** n for i in range(n + 2)) / factorial(n)


def _spline_mass_symbol(theta, degree: int):
    """A_p(θ) = Σ_j N_{2p+1}(p + 1 + j) e^{ijθ}: the symbol of the uniform B-spline mass matrix / h."""
    if degree == 0:
        return np.ones_like(theta)
    n = 2 * degree + 1
    total = _cardinal_bspline(n, degree + 1.0) * np.ones_like(theta)
    for j in range(1, degree + 1):
        total = total + 2 * _cardinal_bspline(n, degree + 1.0 + j) * np.cos(j * theta)
    return total


def spline_galerkin_dispersion(k, dx, degree: int, c=1.0):
    """Compute the numerical frequency of the periodic B-spline Galerkin wave equation in 1-D.

    Galerkin discretization of u_tt = c² u_xx with uniform, maximally smooth B-splines of
    degree p on a periodic grid of spacing dx gives M ü = −c² K u, with the mass and stiffness
    matrices M_ij = ∫ B_i B_j and K_ij = ∫ B_i' B_j'. Both are circulant; their symbols
    (Fourier transforms of a row, θ = k dx) follow from ∫ N_p(x) N_p(x − j) dx = N_{2p+1}(p+1+j)
    for the cardinal B-spline N_p on [0, p + 1]:

        M(θ) = dx A_p(θ),    A_p(θ) = Σ_{|j| ≤ p} N_{2p+1}(p+1+j) cos(jθ)
                                    = Σ_m [sinc((θ + 2πm)/2)]^{2p+2},
        K(θ) = 4 sin²(θ/2) A_{p−1}(θ)/dx,

    the second because N_p' = N_{p−1}(x) − N_{p−1}(x − 1): the derivative of a degree-p spline
    is a degree-(p − 1) spline (the discrete de Rham complex V0 → V1). Hence

        (ω dx/c)² = 4 sin²(θ/2) A_{p−1}(θ)/A_p(θ),

    with A_0 = 1, A_1 = (2 + cos θ)/3, A_2 = (33 + 26 cos θ + cos 2θ)/60. For p = 1 this is the
    linear finite-element result (ω dx/c)² = 6(1 − cos θ)/(2 + cos θ). The relative error is

        ω/(c k) − 1 ≈ |B_{2p}| θ^{2p} / (2 (2p)!)

    (B_n the Bernoulli numbers: θ²/24, θ⁴/1440, θ⁶/60480, θ⁸/2419200 for p = 1, ..., 4): the
    numerical waves are too fast, and there is one branch per wavenumber (no optical modes). The same relation holds for the mixed formulation in Struphy's discrete de
    Rham complex (u in V0, u_x in V1 with the V1 mass matrix), because Dᵀ M1 D = K. ω/c is the
    modified wavenumber of the Galerkin second derivative M⁻¹K.

    Parameters
    ----------
    k : float or array_like
        The wavenumber.
    dx : float or array_like
        The grid spacing (the knot spacing).
    degree : int
        The spline degree p ≥ 1.
    c : float or array_like, optional
        The wave speed. Default: ``1.0``.

    Returns
    -------
    float or numpy.ndarray
        The numerical frequency ω ≥ 0 (semi-discrete: exact in time; see
        :func:`numerical_frequency` for the time integrator's effect).

    Raises
    ------
    ValueError
        If ``degree`` is not a positive integer.

    References
    ----------
    J. A. Cottrell, A. Reali, Y. Bazilevs and T. J. R. Hughes, "Isogeometric analysis of
    structural vibrations", Comput. Methods Appl. Mech. Engrg. 195, 5257 (2006). T. J. R. Hughes,
    A. Reali and G. Sangalli, "Duality and unified analysis of discrete approximations in
    structural dynamics and wave propagation", Comput. Methods Appl. Mech. Engrg. 197, 4104
    (2008). T. J. R. Hughes, "The Finite Element Method", Dover (2000), Ch. 9.

    Examples
    --------
    >>> theta = np.pi / 4
    >>> print(np.round(spline_galerkin_dispersion(theta, 1.0, [1, 2, 3]) / theta, 6))
    [1.025859 1.0003   1.000005]
    """
    degrees = np.asarray(degree)
    if not np.issubdtype(degrees.dtype, np.integer) or np.any(degrees < 1):
        raise ValueError(f"degree must be a positive integer, not {degree!r}")
    k = np.asarray(k, dtype=float)
    dx = np.asarray(dx, dtype=float)
    theta = k * dx
    result = np.zeros(np.broadcast(theta, degrees).shape)
    for p in np.unique(degrees).tolist():
        ratio = _spline_mass_symbol(theta, p - 1) / _spline_mass_symbol(theta, p)
        result = np.where(degrees == p, 2 * np.abs(np.sin(theta / 2)) * np.sqrt(ratio), result)
    return (np.asarray(c, dtype=float) * result / dx)[()]


# ---------------------------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------------------------


