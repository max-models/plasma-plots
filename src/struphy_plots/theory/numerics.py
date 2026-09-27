"""Stability and accuracy of the numerics: CFL numbers, amplification and phase errors of time
integrators, numerical dispersion of finite differences, the Yee scheme and spline Galerkin
discretizations, grid resolution and PIC noise.

Use these to judge a simulation result: is the energy drift, the frequency error or the noise
level of a run what the scheme is expected to produce? All functions are plain numpy and
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
def cfl_number(dt, dx, speed, combine: str = "sum"):
    """Compute the CFL (Courant) number of a time step on a grid.

    In one dimension C = |speed| dt/dx. For several dimensions pass the grid spacings (and
    optionally the speeds) per direction; ``combine`` sets how the directions add up:

    * ``"sum"``: C = Σ |v_i| dt/dx_i, the condition C ≤ 1 of dimensionally unsplit explicit
      upwind / donor-cell advection (the strictest and the safe default).
    * ``"euclidean"``: C = dt √(Σ (v_i/dx_i)²), the Courant condition C ≤ 1 of the Yee /
      leapfrog scheme for Maxwell's equations and the wave equation.
    * ``"max"``: C = max_i |v_i| dt/dx_i, for dimensionally split schemes (each 1-D sweep needs
      C ≤ 1).

    Parameters
    ----------
    dt : float or array_like
        The time step.
    dx : float, array_like or sequence of them
        The grid spacing, or one spacing per direction.
    speed : float, array_like or sequence of them
        The fastest signal speed (e.g. c, the fast magnetosonic speed, the largest marker
        velocity), or one speed per direction.
    combine : {"sum", "euclidean", "max"}, optional
        How the directions combine. Default: ``"sum"``.

    Returns
    -------
    float or numpy.ndarray
        The CFL number.

    Raises
    ------
    ValueError
        If ``combine`` is unknown or the numbers of speeds and spacings differ.

    See Also
    --------
    max_time_step : The largest time step for a given CFL number.

    References
    ----------
    R. Courant, K. Friedrichs and H. Lewy, "Über die partiellen Differenzengleichungen der
    mathematischen Physik", Math. Ann. 100, 32 (1928). R. J. LeVeque, "Finite Volume Methods for
    Hyperbolic Problems", Cambridge University Press (2002), Ch. 4 and 20.

    Examples
    --------
    >>> print(round(float(cfl_number(0.01, 0.05, 2.0)), 4))
    0.4
    >>> print(round(float(cfl_number(0.01, (0.05, 0.1), 2.0)), 4))
    0.6
    >>> print(round(float(cfl_number(0.01, (0.05, 0.1), 2.0, combine="euclidean")), 4))
    0.4472
    """
    return (np.asarray(dt, dtype=float) * _combine(_rates(dx, speed), combine))[()]


def max_time_step(dx, speed, cfl: float = 1.0, combine: str = "sum"):
    """Compute the largest time step for a given CFL number.

    dt_max = cfl / Σ(|v_i|/dx_i) (or the ``"euclidean"`` / ``"max"`` combination, see
    :func:`cfl_number`). With ``combine="euclidean"`` and ``cfl=1`` this is the Courant limit
    dt ≤ 1/(c √(Σ 1/dx_i²)) of the Yee scheme, dx/(c √d) on a uniform grid in d dimensions.

    Parameters
    ----------
    dx : float, array_like or sequence of them
        The grid spacing, or one spacing per direction.
    speed : float, array_like or sequence of them
        The fastest signal speed, or one speed per direction.
    cfl : float or array_like, optional
        The target CFL number. Default: ``1.0``.
    combine : {"sum", "euclidean", "max"}, optional
        How the directions combine. Default: ``"sum"``.

    Returns
    -------
    float or numpy.ndarray
        The largest time step.

    Raises
    ------
    ValueError
        If ``combine`` is unknown or the numbers of speeds and spacings differ.

    See Also
    --------
    cfl_number : The CFL number of a time step.

    References
    ----------
    A. Taflove and S. C. Hagness, "Computational Electrodynamics: The Finite-Difference
    Time-Domain Method", 3rd ed., Artech House (2005), Ch. 4.

    Examples
    --------
    The Courant limit of a 3-D Yee grid with dx = 0.1 and c = 1:

    >>> print(round(float(max_time_step((0.1, 0.1, 0.1), 1.0, combine="euclidean")), 4))
    0.0577
    """
    with np.errstate(divide="ignore"):
        return (np.asarray(cfl, dtype=float) / _combine(_rates(dx, speed), combine))[()]


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


def finite_difference_wavenumber(k, dx, order: int = 2, derivative: int = 1):
    """Compute the modified wavenumber of central finite differences.

    A central difference applied to exp(ikx) returns i k̃ exp(ikx) (first derivative) or
    −k̃² exp(ikx) (second derivative). With θ = k dx:

    * first derivative, order 2: k̃ dx = sin θ; order 4: (8 sin θ − sin 2θ)/6; order 6:
      (45 sin θ − 9 sin 2θ + sin 3θ)/30. The relative error k̃/k − 1 is −θ²/6, −θ⁴/30 and
      −θ⁶/140 to leading order.
    * second derivative, order 2: (k̃ dx)² = 2 − 2 cos θ = 4 sin²(θ/2); order 4:
      [16(2 − 2 cos θ) − (2 − 2 cos 2θ)]/12; order 6:
      [270(2 − 2 cos θ) − 27(2 − 2 cos 2θ) + 2(2 − 2 cos 3θ)]/180.

    The numerical phase velocity of advection u_t + a u_x = 0 with a first-derivative stencil
    (and exact time integration) is a k̃/k; the frequency of the wave equation with a
    second-derivative stencil is c k̃.

    Parameters
    ----------
    k : float or array_like
        The wavenumber.
    dx : float or array_like
        The grid spacing.
    order : {2, 4, 6}, optional
        The order of accuracy of the stencil. Default: ``2``.
    derivative : {1, 2}, optional
        The derivative the stencil approximates. Default: ``1``.

    Returns
    -------
    float or numpy.ndarray
        The modified wavenumber k̃, with the sign of ``k``.

    Raises
    ------
    ValueError
        If ``order`` or ``derivative`` is not supported.

    References
    ----------
    S. K. Lele, "Compact finite difference schemes with spectral-like resolution",
    J. Comput. Phys. 103, 16 (1992). B. Fornberg, "Generation of finite difference formulas on
    arbitrarily spaced grids", Math. Comp. 51, 699 (1988).

    Examples
    --------
    >>> print(np.round(finite_difference_wavenumber(np.pi / 2, 1.0, order=[2, 4, 6]), 4))
    [1.     1.3333 1.4667]
    """
    table = {1: _FD_FIRST, 2: _FD_SECOND}.get(derivative)
    if table is None:
        raise ValueError(f"derivative must be 1 or 2, not {derivative!r}")
    k = np.asarray(k, dtype=float)
    dx = np.asarray(dx, dtype=float)
    orders = np.asarray(order)
    unknown = set(np.unique(orders).tolist()) - set(table)
    if unknown:
        raise ValueError(f"order must be 2, 4 or 6, not {sorted(unknown)}")
    theta = k * dx
    result = np.zeros(np.broadcast(theta, orders).shape)
    for n, coefficients in table.items():
        if derivative == 1:
            value = sum(a * np.sin((j + 1) * theta) for j, a in enumerate(coefficients))
        else:
            value = np.sign(theta) * np.sqrt(
                sum(b * (2 - 2 * np.cos((j + 1) * theta)) for j, b in enumerate(coefficients))
            )
        result = np.where(orders == n, value, result)
    return (result / dx)[()]


def yee_dispersion(k, dx, dt, c=1.0, dims: int = 1):
    """Compute the numerical frequency of the Yee (leapfrog) scheme for light waves.

    The staggered leapfrog in space and time for Maxwell's equations (or the second-order wave
    equation) gives

        sin²(ω dt/2)/(c dt)² = Σ_i sin²(k_i dx_i/2)/dx_i²,

    so in one dimension ω = (2/dt) arcsin((c dt/dx) sin(k dx/2)), exact (ω = c|k|) at the "magic
    time step" c dt = dx. When the right-hand side times (c dt)² exceeds 1 (beyond the Courant
    limit c dt ≤ 1/√(Σ 1/dx_i²)), ω = (2/dt)(π/2 + i arccosh(s)) is complex with a positive
    growth rate. Returns the positive root.

    Parameters
    ----------
    k : float, array_like or sequence of them
        The wavenumber: a magnitude, or the components (k_1, ..., k_d). A magnitude with
        ``dims`` > 1 propagates along the grid diagonal, k_i = k/√dims, the direction of the
        least stable mode (the Brillouin-zone corner).
    dx : float, array_like or sequence of them
        The grid spacing, or one spacing per direction.
    dt : float or array_like
        The time step.
    c : float or array_like, optional
        The wave speed. Default: ``1.0``.
    dims : int, optional
        The number of dimensions when ``k`` is a magnitude. Default: ``1``, or the number of
        components of ``k`` / ``dx``.

    Returns
    -------
    complex or numpy.ndarray
        The numerical frequency ω, complex.

    Raises
    ------
    ValueError
        If the numbers of components of ``k`` and ``dx`` differ.

    See Also
    --------
    max_time_step : The Courant limit, with ``combine="euclidean"``.

    References
    ----------
    K. S. Yee, "Numerical solution of initial boundary value problems involving Maxwell's
    equations in isotropic media", IEEE Trans. Antennas Propag. 14, 302 (1966). A. Taflove and
    S. C. Hagness, "Computational Electrodynamics: The Finite-Difference Time-Domain Method",
    3rd ed., Artech House (2005), Ch. 4. C. K. Birdsall and A. B. Langdon, "Plasma Physics via
    Computer Simulation", IOP Publishing (1991), Ch. 15.

    Examples
    --------
    >>> print(np.round(yee_dispersion(1.0, 0.5, 0.25).real, 5))
    0.99216
    >>> print(np.round(yee_dispersion(1.0, 0.5, 0.5).real, 5))
    1.0
    """
    ks = _components(k)
    dxs = _components(dx)
    d = max(len(ks), len(dxs), int(dims))
    if len(ks) == 1 and d > 1:
        ks = [ks[0] / np.sqrt(d)] * d
    if len(dxs) == 1:
        dxs = dxs * d
    if len(ks) != len(dxs):
        raise ValueError(f"got {len(ks)} wavevector components for {len(dxs)} grid spacings")
    dt = np.asarray(dt, dtype=float)
    s = np.asarray(c, dtype=float) * dt * np.sqrt(sum((np.sin(kk * h / 2) / h) ** 2 for kk, h in zip(ks, dxs)))
    stable = s <= 1
    omega = np.where(
        stable,
        2 / dt * np.arcsin(np.minimum(s, 1.0)) + 0j,
        2 / dt * (np.pi / 2 + 1j * np.arccosh(np.maximum(s, 1.0))),
    )
    return omega[()]


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


def points_per_wavelength(k, dx):
    """Compute the number of grid points per wavelength, 2π/(|k| dx).

    Two points per wavelength is the Nyquist limit (k dx = π); second-order schemes need roughly
    10–20 for percent-level phase accuracy, higher-order splines far fewer.

    Parameters
    ----------
    k : float or array_like
        The wavenumber.
    dx : float or array_like
        The grid spacing.

    Returns
    -------
    float or numpy.ndarray
        Points per wavelength (``inf`` for k = 0).

    References
    ----------
    A. Taflove and S. C. Hagness, "Computational Electrodynamics: The Finite-Difference
    Time-Domain Method", 3rd ed., Artech House (2005), Ch. 4.

    Examples
    --------
    >>> print(round(float(points_per_wavelength(2 * np.pi, 0.05)), 4))
    20.0
    """
    with np.errstate(divide="ignore"):
        return (2 * np.pi / (np.abs(np.asarray(k, dtype=float)) * np.asarray(dx, dtype=float)))[()]


# ---------------------------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------------------------
def pic_noise(markers, weight_rms=1.0):
    """Estimate the relative statistical noise of a particle (Monte Carlo) estimate.

    A moment (density, current, a cell average of the distribution function) estimated from N
    independent markers has the relative standard deviation σ/⟨·⟩ = w_rms/√N. For the noise of a
    grid quantity, N is the number of markers per cell (per basis-function support for
    spline deposition, which smooths over p + 1 cells per direction and so lowers the noise); for
    a global quantity (total energy, a Fourier mode amplitude) the total number of markers.
    ``weight_rms`` is the root-mean-square marker weight relative to the mean weight of a full-f
    run with equal weights: 1 for full-f, √(⟨w²⟩)/⟨w⟩ for unequal weights, and for δf the rms of
    the δf weights, which then gives the noise relative to the background.

    Parameters
    ----------
    markers : float or array_like
        The number of markers N contributing to the estimate (e.g. markers per cell).
    weight_rms : float or array_like, optional
        The rms marker weight relative to equal full-f weights. Default: ``1.0``.

    Returns
    -------
    float or numpy.ndarray
        The relative noise w_rms/√N.

    See Also
    --------
    markers_for_noise : The inverse.

    References
    ----------
    C. K. Birdsall and A. B. Langdon, "Plasma Physics via Computer Simulation", IOP Publishing
    (1991), Ch. 12. R. Hatzky, T. M. Tran, A. Könies, R. Kleiber and S. J. Allfrey, "Energy
    conservation in a nonlinear gyrokinetic particle-in-cell code for ion-temperature-gradient-
    driven modes in θ-pinch geometry", Phys. Plasmas 9, 898 (2002).

    Examples
    --------
    >>> print(np.round(pic_noise([100, 10_000]), 4))
    [0.1  0.01]
    """
    return (np.asarray(weight_rms, dtype=float) / np.sqrt(np.asarray(markers, dtype=float)))[()]


def markers_for_noise(target_noise, weight_rms=1.0):
    """Compute the number of markers needed for a relative noise level, N = (w_rms/noise)².

    Parameters
    ----------
    target_noise : float or array_like
        The relative noise wanted (e.g. 0.01 for 1 %).
    weight_rms : float or array_like, optional
        The rms marker weight relative to equal full-f weights; see :func:`pic_noise`.
        Default: ``1.0``.

    Returns
    -------
    float or numpy.ndarray
        The number of markers (per cell, or in total; see :func:`pic_noise`), not rounded.

    See Also
    --------
    pic_noise : The noise of N markers.

    References
    ----------
    C. K. Birdsall and A. B. Langdon, "Plasma Physics via Computer Simulation", IOP Publishing
    (1991), Ch. 12.

    Examples
    --------
    A δf run with rms weight 0.1 needs 100 markers per cell for 1 % noise:

    >>> print(round(float(markers_for_noise(0.01, weight_rms=0.1)), 6))
    100.0
    """
    return ((np.asarray(weight_rms, dtype=float) / np.asarray(target_noise, dtype=float)) ** 2)[()]


def debye_resolution(dx, debye_length):
    """Compute the grid spacing in Debye lengths, dx/λ_D.

    Parameters
    ----------
    dx : float or array_like
        The grid spacing.
    debye_length : float or array_like
        The Debye length λ_D, in the same units (1 in Struphy's kinetic normalization).

    Returns
    -------
    float or numpy.ndarray
        dx/λ_D.

    See Also
    --------
    finite_grid_stable : The finite-grid instability criterion.

    References
    ----------
    C. K. Birdsall and A. B. Langdon, "Plasma Physics via Computer Simulation", IOP Publishing
    (1991), Ch. 8.

    Examples
    --------
    >>> print(round(float(debye_resolution(0.5, 1.0)), 4))
    0.5
    """
    return (np.asarray(dx, dtype=float) / np.asarray(debye_length, dtype=float))[()]


def finite_grid_stable(dx, debye_length, max_ratio=np.pi):
    """Check the finite-grid (aliasing) instability criterion of momentum-conserving PIC.

    In momentum-conserving PIC (NGP or linear/CIC weighting with a gridded field), a thermal
    plasma whose Debye length is unresolved heats up through aliasing of the grid modes.
    Birdsall and Langdon find the instability negligible for λ_D ≳ dx/π with linear weighting,
    i.e. dx/λ_D ≲ π (the default ``max_ratio``); the common safe practice is dx ≲ λ_D
    (``max_ratio=1``). Higher-order shapes weaken the instability further. Energy-conserving
    schemes (implicit or structure-preserving particle–field couplings, like Struphy's) do not
    have this instability, but dx ≫ λ_D still under-resolves Debye shielding and the
    short-wavelength physics.

    Parameters
    ----------
    dx : float or array_like
        The grid spacing.
    debye_length : float or array_like
        The Debye length λ_D.
    max_ratio : float or array_like, optional
        The largest allowed dx/λ_D. Default: ``π``.

    Returns
    -------
    bool or numpy.ndarray
        ``True`` where dx/λ_D ≤ ``max_ratio``.

    See Also
    --------
    debye_resolution : The ratio dx/λ_D.

    References
    ----------
    C. K. Birdsall and A. B. Langdon, "Plasma Physics via Computer Simulation", IOP Publishing
    (1991), Ch. 8 and 12. A. B. Langdon, "Effects of the spatial grid in simulation
    plasmas", J. Comput. Phys. 6, 247 (1970). H. R. Lewis, "Energy-conserving numerical
    approximations for Vlasov plasmas", J. Comput. Phys. 6, 136 (1970).

    Examples
    --------
    >>> print(finite_grid_stable([0.5, 2.0, 5.0], 1.0))
    [ True  True False]
    """
    return (debye_resolution(dx, debye_length) <= np.asarray(max_ratio, dtype=float))[()]
