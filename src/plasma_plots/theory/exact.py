"""Exact solutions to verify simulations: the Riemann problem of gas dynamics (with vacuum), the
dam break, diffusion, advection, the wave equation, and pressureless flow and its caustics.

Profiles take the coordinates first and the time second, ``f(x, t, ...)``, so that
``lambda x, t: f(x, t, ...)`` (or a field of the returned result) can be passed as the
``reference=`` of plasma-plots' profile plots. Everything is plain numpy and vectorized: the
coordinates and times broadcast against each other.

Where a state can be empty (a vacuum in gas dynamics, a dry bed in shallow water) its density
or depth is zero and its velocity is ``nan`` (undefined, so plots leave a gap); momenta and
energies there are zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "LagrangianFlow",
    "RiemannSolution",
    "ShallowWaterSolution",
    "StarState",
    "advected",
    "caustic_time",
    "dalembert",
    "dam_break",
    "heat_kernel",
    "pressureless",
    "pressureless_eulerian",
    "riemann_euler",
    "sod_shock_tube",
    "star_state",
]


def _out(a):
    """A numpy array, or a numpy scalar for 0-d input."""
    return np.asarray(a)[()]


def _similarity(x, t, x0):
    """The similarity variable (x − x0)/t, with ∓∞ at t = 0 (the initial discontinuity)."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if np.any(t < 0):
        raise ValueError("t must be non-negative")
    x, t = np.broadcast_arrays(x - x0, t)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, x / np.where(t > 0, t, 1.0), np.where(x < 0, -np.inf, np.inf))


# ---------------------------------------------------------------------------------------------
# The Riemann problem of the Euler equations
# ---------------------------------------------------------------------------------------------
@dataclass
class StarState:
    """The star region of an Euler Riemann problem, between the left and the right wave.

    Attributes
    ----------
    pressure : float
        The star pressure p* (zero if a vacuum forms).
    velocity : float
        The star (contact) velocity u*; ``nan`` if a vacuum forms or a side is a vacuum.
    density_left : float
        The density ρ*L left of the contact (zero for a vacuum).
    density_right : float
        The density ρ*R right of the contact (zero for a vacuum).
    left_wave : str
        ``"shock"`` or ``"rarefaction"`` (``"none"`` if the left state is a vacuum).
    right_wave : str
        ``"shock"`` or ``"rarefaction"`` (``"none"`` if the right state is a vacuum).
    vacuum : bool
        Whether the solution contains a vacuum (given, or generated between two rarefactions).
    """

    pressure: float
    velocity: float
    density_left: float
    density_right: float
    left_wave: str
    right_wave: str
    vacuum: bool


@dataclass
class RiemannSolution:
    """The primitive variables of a gas-dynamics solution.

    Attributes
    ----------
    density : float or numpy.ndarray
        The mass density ρ.
    velocity : float or numpy.ndarray
        The velocity u; ``nan`` in a vacuum.
    pressure : float or numpy.ndarray
        The pressure p.
    internal_energy : float or numpy.ndarray
        The specific internal energy e = p/((γ − 1) ρ); ``nan`` in a vacuum.
    gamma : float
        The ratio of specific heats γ.
    """

    density: object
    velocity: object
    pressure: object
    internal_energy: object
    gamma: float

    @property
    def momentum(self):
        """The momentum density ρu (zero in a vacuum)."""
        return _out(
            np.where(
                np.asarray(self.density) > 0,
                np.asarray(self.density) * self.velocity,
                0.0,
            )
        )

    @property
    def energy(self):
        """The total energy density ρu²/2 + p/(γ − 1) (zero in a vacuum)."""
        rho = np.asarray(self.density)
        kinetic = np.where(rho > 0, 0.5 * rho * np.where(rho > 0, self.velocity, 0.0) ** 2, 0.0)
        return _out(kinetic + np.asarray(self.pressure) / (self.gamma - 1))

    @property
    def sound_speed(self):
        """The sound speed a = √(γ p/ρ); ``nan`` in a vacuum."""
        rho = np.asarray(self.density)
        with np.errstate(divide="ignore", invalid="ignore"):
            return _out(
                np.where(
                    rho > 0,
                    np.sqrt(self.gamma * np.asarray(self.pressure) / rho),
                    np.nan,
                )
            )


def _check_state(state, side):
    rho, u, p = (float(v) for v in state)
    if rho < 0 or p < 0:
        raise ValueError(f"the {side} density and pressure must be non-negative; got {state!r}")
    if (rho == 0) != (p == 0):
        raise ValueError(
            f"the {side} state must have both density and pressure zero (a vacuum) or both positive; got {state!r}"
        )
    return rho, u, p


def _pressure_function(p, rho, pk, gamma):
    """Toro's f_K(p) and its derivative for one side (4.6, 4.7)."""
    a = np.sqrt(gamma * pk / rho)
    if p > pk:  # shock
        big_a = 2 / ((gamma + 1) * rho)
        big_b = (gamma - 1) / (gamma + 1) * pk
        root = np.sqrt(big_a / (big_b + p))
        return (p - pk) * root, root * (1 - (p - pk) / (2 * (big_b + p)))
    ratio = p / pk
    f = 2 * a / (gamma - 1) * (ratio ** ((gamma - 1) / (2 * gamma)) - 1)
    return f, ratio ** (-(gamma + 1) / (2 * gamma)) / (rho * a)


def _solve_star_pressure(left, right, gamma):
    """Newton's method, safeguarded by bisection, for f_L(p) + f_R(p) + u_R − u_L = 0."""
    (rl, ul, pl), (rr, ur, pr) = left, right
    du = ur - ul

    def f(p):
        fl, dl = _pressure_function(p, rl, pl, gamma)
        fr, dr = _pressure_function(p, rr, pr, gamma)
        return fl + fr + du, dl + dr, fl, fr

    al, ar = np.sqrt(gamma * pl / rl), np.sqrt(gamma * pr / rr)
    z = (gamma - 1) / (2 * gamma)
    # the two-rarefaction guess (exact if both waves are rarefactions), Toro (4.46)
    guess = ((al + ar - 0.5 * (gamma - 1) * du) / (al / pl**z + ar / pr**z)) ** (1 / z)
    low, high = 0.0, max(pl, pr, guess)
    while f(high)[0] < 0:
        high *= 2
    p = min(max(guess, 1e-300), high)
    for _ in range(200):
        value, slope, _, _ = f(p)
        if value < 0:
            low = p
        else:
            high = p
        new = p - value / slope
        if not low < new < high:
            new = 0.5 * (low + high)
        if abs(new - p) <= 1e-15 * max(new, 1e-300) or high - low <= 1e-15 * high:
            p = new
            break
        p = new
    _, _, fl, fr = f(p)
    return p, 0.5 * (ul + ur) + 0.5 * (fr - fl)


def star_state(left, right, gamma=1.4):
    """Solve for the star region of the Riemann problem of the 1-D Euler equations.

    The star pressure p* is the root of Toro's pressure function f_L(p) + f_R(p) + u_R − u_L,
    found by Newton's method safeguarded by bisection to machine precision. A wave is a shock
    if p* exceeds the pressure ahead of it and a rarefaction otherwise. If a side is a vacuum
    (density and pressure zero), or if 2(a_L + a_R)/(γ − 1) ≤ u_R − u_L so that the two
    rarefactions leave a vacuum between them, p* = 0.

    Parameters
    ----------
    left : (float, float, float)
        The left state (density, velocity, pressure).
    right : (float, float, float)
        The right state (density, velocity, pressure).
    gamma : float, optional
        The ratio of specific heats γ > 1. Default: ``1.4``.

    Returns
    -------
    StarState
        p*, u*, the densities on both sides of the contact and the types of the two waves.

    Raises
    ------
    ValueError
        If ``gamma`` ≤ 1, or a state has a negative density or pressure, or only one of the
        two zero.

    See Also
    --------
    riemann_euler : The whole solution.

    References
    ----------
    E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. (Springer,
    2009), chapter 4.

    Examples
    --------
    Toro's test 1 (Sod's shock tube):

    >>> s = star_state((1.0, 0.0, 1.0), (0.125, 0.0, 0.1))
    >>> (
    ...     round(s.pressure, 5),
    ...     round(s.velocity, 5),
    ...     round(s.density_left, 5),
    ...     round(s.density_right, 5),
    ... )
    (0.30313, 0.92745, 0.42632, 0.26557)
    >>> s.left_wave, s.right_wave
    ('rarefaction', 'shock')
    """
    if gamma <= 1:
        raise ValueError(f"gamma must be larger than 1; got {gamma!r}")
    rl, ul, pl = _check_state(left, "left")
    rr, ur, pr = _check_state(right, "right")
    if rl == 0 or rr == 0:
        return StarState(
            0.0,
            np.nan,
            0.0,
            0.0,
            "none" if rl == 0 else "rarefaction",
            "none" if rr == 0 else "rarefaction",
            True,
        )
    al, ar = np.sqrt(gamma * pl / rl), np.sqrt(gamma * pr / rr)
    if 2 * (al + ar) / (gamma - 1) <= ur - ul:
        return StarState(0.0, np.nan, 0.0, 0.0, "rarefaction", "rarefaction", True)
    p, u = _solve_star_pressure((rl, ul, pl), (rr, ur, pr), gamma)
    g6 = (gamma - 1) / (gamma + 1)

    def density(rho, pk):
        if p > pk:
            return rho * (p / pk + g6) / (g6 * p / pk + 1)
        return rho * (p / pk) ** (1 / gamma)

    return StarState(
        float(p),
        float(u),
        float(density(rl, pl)),
        float(density(rr, pr)),
        "shock" if p > pl else "rarefaction",
        "shock" if p > pr else "rarefaction",
        False,
    )


def _left_wave(s, rho, u, p, p_star, u_star, gamma):
    """Sample the left wave and the left star state at speeds s (Toro 4.5); p_star = 0 is a vacuum."""
    a = np.sqrt(gamma * p / rho)
    if p_star > p:  # shock
        g6 = (gamma - 1) / (gamma + 1)
        rho_star = rho * (p_star / p + g6) / (g6 * p_star / p + 1)
        speed = u - a * np.sqrt((gamma + 1) / (2 * gamma) * p_star / p + (gamma - 1) / (2 * gamma))
        ahead = s < speed
        return (
            np.where(ahead, rho, rho_star),
            np.where(ahead, u, u_star),
            np.where(ahead, p, p_star),
        )
    ratio = p_star / p
    rho_star = rho * ratio ** (1 / gamma)
    head, tail = u - a, u_star - a * ratio ** ((gamma - 1) / (2 * gamma))
    with np.errstate(invalid="ignore", over="ignore"):
        base = np.clip(2 / (gamma + 1) + (gamma - 1) / ((gamma + 1) * a) * (u - s), 0.0, None)
        fan = (
            rho * base ** (2 / (gamma - 1)),
            2 / (gamma + 1) * (a + 0.5 * (gamma - 1) * u + s),
            p * base ** (2 * gamma / (gamma - 1)),
        )
    ahead, behind = s < head, s > tail
    return tuple(
        np.where(ahead, w, np.where(behind, w_star, w_fan))
        for w, w_star, w_fan in zip((rho, u, p), (rho_star, u_star, p_star), fan)
    )


def _right_wave(s, rho, u, p, p_star, u_star, gamma):
    """The right wave, as the mirror image x → −x of a left wave."""
    r, v, q = _left_wave(-s, rho, -u, p, p_star, -u_star, gamma)
    return r, -v, q


def riemann_euler(x, t, left, right, gamma=1.4, x0=0.0):
    """Compute the exact solution of the Riemann problem of the 1-D Euler equations.

    The gas is ideal, p = (γ − 1) ρ e, and initially in the state ``left`` for x < x0 and
    ``right`` for x > x0. The solution is self-similar in (x − x0)/t: a left wave (shock or
    rarefaction), a contact discontinuity moving at u* and a right wave. All configurations are
    covered, including a vacuum on either side (gas expanding into vacuum, bounded by a front
    moving at u ± 2a/(γ − 1)) and the vacuum generated between two strong rarefactions.

    Parameters
    ----------
    x : float or array_like
        Positions.
    t : float or array_like
        Times ≥ 0, broadcast against ``x``. At t = 0 the initial states are returned.
    left : (float, float, float)
        The left state (density, velocity, pressure); ``(0, u, 0)`` is a vacuum.
    right : (float, float, float)
        The right state (density, velocity, pressure); ``(0, u, 0)`` is a vacuum.
    gamma : float, optional
        The ratio of specific heats γ > 1. Default: ``1.4``.
    x0 : float, optional
        The position of the initial discontinuity. Default: ``0.0``.

    Returns
    -------
    RiemannSolution
        Density, velocity, pressure and specific internal energy, with the broadcast shape of
        ``x`` and ``t``; also the momentum and energy densities as properties.

    Raises
    ------
    ValueError
        If ``gamma`` ≤ 1, a time is negative, or a state is invalid (see :func:`star_state`).

    See Also
    --------
    star_state : The star pressure and velocity alone.
    sod_shock_tube : The classic test case.

    References
    ----------
    E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics, 3rd ed. (Springer,
    2009), chapter 4 (4.9 for vacuum).

    Examples
    --------
    Toro's test 1 at t = 0.25, left of, inside and right of the star region:

    >>> w = riemann_euler(
    ...     [-0.4, 0.1, 0.5], 0.25, (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)
    ... )
    >>> w.density.round(5), w.pressure.round(5)
    (array([1.     , 0.42632, 0.125  ]), array([1.     , 0.30313, 0.1    ]))

    Gas expanding into vacuum: the front moves at 2a/(γ − 1) = 5.9161 here.

    >>> w = riemann_euler([0.0, 3.0, 6.0], 1.0, (1.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    >>> w.density.round(5), w.velocity.round(5)
    (array([0.40188, 0.01169, 0.     ]), array([0.98601, 3.48601,     nan]))
    """
    s = _similarity(x, t, x0)
    star = star_state(left, right, gamma)
    rl, ul, pl = (float(v) for v in left)
    rr, ur, pr = (float(v) for v in right)
    vacuum = (np.zeros_like(s), np.zeros_like(s), np.zeros_like(s))
    if rl == 0 and rr == 0:
        rho, u, p = vacuum
    elif rr == 0:
        front = ul + 2 * np.sqrt(gamma * pl / rl) / (gamma - 1)
        rho, u, p = _left_wave(s, rl, ul, pl, 0.0, front, gamma)
    elif rl == 0:
        front = ur - 2 * np.sqrt(gamma * pr / rr) / (gamma - 1)
        rho, u, p = _right_wave(s, rr, ur, pr, 0.0, front, gamma)
    else:
        if star.vacuum:
            u_left = ul + 2 * np.sqrt(gamma * pl / rl) / (gamma - 1)
            u_right = ur - 2 * np.sqrt(gamma * pr / rr) / (gamma - 1)
        else:
            u_left = u_right = star.velocity
        wl = _left_wave(s, rl, ul, pl, star.pressure, u_left, gamma)
        wr = _right_wave(s, rr, ur, pr, star.pressure, u_right, gamma)
        rho, u, p = (np.where(s < u_left, a, b) for a, b in zip(wl, wr))
    empty = rho <= 0
    u = np.where(empty, np.nan, u)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(empty, np.nan, p / ((gamma - 1) * np.where(empty, 1.0, rho)))
    return RiemannSolution(
        _out(np.where(empty, 0.0, rho)),
        _out(u),
        _out(np.where(empty, 0.0, p)),
        _out(e),
        float(gamma),
    )


def sod_shock_tube(x, t, gamma=1.4, x0=0.5):
    """Compute the exact solution of Sod's shock tube.

    The Riemann problem with (ρ, u, p) = (1, 0, 1) left and (0.125, 0, 0.1) right of x0, usually
    on [0, 1] up to t = 0.2 (Toro's test 1): a left rarefaction, a contact and a right shock.

    Parameters
    ----------
    x : float or array_like
        Positions.
    t : float or array_like
        Times ≥ 0, broadcast against ``x``.
    gamma : float, optional
        The ratio of specific heats. Default: ``1.4``.
    x0 : float, optional
        The position of the diaphragm. Default: ``0.5``.

    Returns
    -------
    RiemannSolution
        Density, velocity, pressure and specific internal energy.

    See Also
    --------
    riemann_euler : Any Riemann problem.

    References
    ----------
    G. A. Sod, "A survey of several finite difference methods for systems of nonlinear
    hyperbolic conservation laws", J. Comput. Phys. 27, 1 (1978).

    Examples
    --------
    >>> w = sod_shock_tube([0.1, 0.6, 0.8, 0.9], 0.2)
    >>> w.density.round(5)
    array([1.     , 0.42632, 0.26557, 0.125  ])
    >>> float(w.velocity[2].round(5)), float(w.pressure[2].round(5))
    (0.92745, 0.30313)
    """
    return riemann_euler(x, t, (1.0, 0.0, 1.0), (0.125, 0.0, 0.1), gamma=gamma, x0=x0)


# ---------------------------------------------------------------------------------------------
# Dam break
# ---------------------------------------------------------------------------------------------
@dataclass
class ShallowWaterSolution:
    """The depth and velocity of a shallow-water solution.

    Attributes
    ----------
    depth : float or numpy.ndarray
        The water depth h (zero on a dry bed).
    velocity : float or numpy.ndarray
        The depth-averaged velocity u; ``nan`` on a dry bed.
    """

    depth: object
    velocity: object

    @property
    def discharge(self):
        """The discharge (momentum per unit density) hu, zero on a dry bed."""
        h = np.asarray(self.depth)
        return _out(np.where(h > 0, h * np.where(h > 0, self.velocity, 0.0), 0.0))


def dam_break(x, t, depth=1.0, gravity=1.0, x0=0.0):
    """Compute Ritter's exact dam-break solution of the 1-D shallow-water equations on a dry bed.

    Water at rest of depth h0 for x < x0 is released at t = 0 onto a dry bed. A rarefaction runs
    between x0 − c0 t and the front at x0 + 2 c0 t, where c0 = √(g h0), with h = (2c0 − ξ)²/(9g)
    and u = 2(c0 + ξ)/3 for ξ = (x − x0)/t.

    Parameters
    ----------
    x : float or array_like
        Positions.
    t : float or array_like
        Times ≥ 0, broadcast against ``x``.
    depth : float, optional
        The initial depth h0 behind the dam (x < x0). Default: ``1.0``.
    gravity : float, optional
        The gravitational acceleration g. Default: ``1.0``.
    x0 : float, optional
        The position of the dam. Default: ``0.0``.

    Returns
    -------
    ShallowWaterSolution
        Depth and velocity (``nan`` on the dry bed), with the broadcast shape of ``x`` and
        ``t``; also the discharge hu as a property.

    Raises
    ------
    ValueError
        If a time is negative, or the depth not positive.

    References
    ----------
    A. Ritter, "Die Fortpflanzung der Wasserwellen", Z. Ver. Dtsch. Ing. 36, 947 (1892).

    Examples
    --------
    >>> w = dam_break([-2.0, 0.0, 1.0, 2.1], 1.0)
    >>> w.depth.round(6), w.velocity.round(6)
    (array([1.      , 0.444444, 0.111111, 0.      ]), array([0.      , 0.666667, 1.333333,      nan]))
    """
    if not depth > 0:
        raise ValueError(f"the depth must be positive; got {depth!r}")
    xi = _similarity(x, t, x0)
    c0 = np.sqrt(gravity * depth)
    with np.errstate(invalid="ignore"):
        h_fan, u_fan = (2 * c0 - xi) ** 2 / (9 * gravity), 2 * (c0 + xi) / 3
    h = np.select([xi < -c0, xi < 2 * c0], [depth, h_fan], 0.0)
    u = np.select([xi < -c0, xi < 2 * c0], [0.0, u_fan], np.nan)
    return ShallowWaterSolution(_out(h), _out(u))


# ---------------------------------------------------------------------------------------------
# Diffusion, advection, waves
# ---------------------------------------------------------------------------------------------
def _coordinates(x):
    """A list of coordinate arrays: a tuple is one array per dimension, anything else is 1-D."""
    return [np.asarray(c, dtype=float) for c in x] if isinstance(x, tuple) else [np.asarray(x, dtype=float)]


def heat_kernel(x, t, diffusivity, width=0.0, center=0.0, mass=1.0):
    """Compute a spreading Gaussian, the solution of the diffusion equation ∂u/∂t = D ∇²u.

    u = M (2π σ²)^(−d/2) exp(−|x − x_c|²/(2σ²)) with σ² = σ0² + 2 D t: the fundamental solution
    for σ0 = 0, and a Gaussian of initial standard deviation σ0 otherwise. The integral is M at
    all times; each coordinate's variance grows as σ0² + 2Dt.

    Parameters
    ----------
    x : float, array_like or tuple of array_like
        Positions: an array in 1-D, or a tuple ``(x, y)`` or ``(x, y, z)`` of arrays that
        broadcast against each other in d dimensions.
    t : float or array_like
        Times ≥ 0, broadcast against the positions.
    diffusivity : float
        The diffusion coefficient D.
    width : float, optional
        The initial standard deviation σ0. Default: ``0.0`` (a point source at t = 0).
    center : float or tuple of float, optional
        The center x_c, one value per dimension (a scalar is used for all). Default: ``0.0``.
    mass : float, optional
        The integral M of u. Default: ``1.0``.

    Returns
    -------
    float or numpy.ndarray
        u; at t = 0 with σ0 = 0, ``inf`` at the center and 0 elsewhere.

    Raises
    ------
    ValueError
        If a time or ``diffusivity`` or ``width`` is negative.

    References
    ----------
    L. C. Evans, Partial Differential Equations, 2nd ed. (AMS, 2010), section 2.3.

    Examples
    --------
    >>> round(float(heat_kernel(0.0, 0.5, 1.0)), 6)  # 1/√(4π D t)
    0.398942
    >>> round(float(heat_kernel((1.0, 0.0), 0.25, 1.0, width=1.0)), 6)
    0.076026
    """
    t = np.asarray(t, dtype=float)
    if np.any(t < 0) or diffusivity < 0 or width < 0:
        raise ValueError("t, diffusivity and width must be non-negative")
    coordinates = _coordinates(x)
    centers = np.broadcast_to(np.asarray(center, dtype=float), (len(coordinates),))
    r2 = sum((c - c0) ** 2 for c, c0 in zip(coordinates, centers))
    variance = width**2 + 2 * diffusivity * t
    r2, variance = np.broadcast_arrays(r2, variance)
    d = len(coordinates)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = mass * np.exp(-r2 / (2 * variance)) / (2 * np.pi * variance) ** (d / 2)
    u = np.where(variance > 0, u, np.where(r2 == 0, np.inf, 0.0))
    return _out(u)


def advected(profile, x, t, velocity, period=None):
    """Compute a profile carried unchanged at constant velocity, u(x, t) = u0(x − v t).

    Parameters
    ----------
    profile : callable
        The initial profile u0, a vectorized function of the position.
    x : float or array_like
        Positions.
    t : float or array_like
        Times, broadcast against ``x``.
    velocity : float
        The advection velocity v.
    period : float or (float, float), optional
        For a periodic domain: its length L (the domain is [0, L)) or its bounds (a, b). The
        shifted position is wrapped into the domain before ``profile`` is called. Default:
        ``None`` (not periodic).

    Returns
    -------
    float or numpy.ndarray
        u0(x − v t).

    Examples
    --------
    >>> # −0.25 wrapped into [0, 1)
    >>> round(float(advected(lambda x: x, 0.1, 0.35, 1.0, period=1.0)), 6)
    0.75
    >>> advected(lambda x: np.exp(-(x**2)), [0.0, 1.0], 1.0, 1.0).round(4)
    array([0.3679, 1.    ])
    """
    shifted = np.asarray(x, dtype=float) - velocity * np.asarray(t, dtype=float)
    if period is not None:
        a, b = (0.0, float(period)) if np.ndim(period) == 0 else (float(period[0]), float(period[1]))
        shifted = a + np.mod(shifted - a, b - a)
    return _out(profile(shifted))


_GAUSS_LEGENDRE = np.polynomial.legendre.leggauss(64)


def dalembert(x, t, initial, speed, initial_rate=None):
    """Compute d'Alembert's solution of the 1-D wave equation ∂²u/∂t² = c² ∂²u/∂x².

    u = [u0(x − ct) + u0(x + ct)]/2 + (1/(2c)) ∫_{x−ct}^{x+ct} v0(s) ds, for the initial profile
    u0 and the initial rate v0 = ∂u/∂t at t = 0, on the whole line. The integral is done by
    64-point Gauss–Legendre quadrature, exact to rounding for smooth v0 over a few wavelengths.
    A pulse at rest splits into two halves moving at ±c.

    Parameters
    ----------
    x : float or array_like
        Positions.
    t : float or array_like
        Times, broadcast against ``x``.
    initial : callable
        The initial profile u0, a vectorized function of the position.
    speed : float
        The wave speed c > 0.
    initial_rate : callable, optional
        The initial rate v0, a vectorized function of the position. Default: ``None`` (at rest).

    Returns
    -------
    float or numpy.ndarray
        u(x, t).

    Raises
    ------
    ValueError
        If ``speed`` is not positive.

    References
    ----------
    J. le Rond d'Alembert, "Recherches sur la courbe que forme une corde tenduë mise en
    vibration", Hist. Acad. R. Sci. Berlin 3, 214 (1747); L. C. Evans, Partial Differential
    Equations, 2nd ed. (AMS, 2010), section 2.4.

    Examples
    --------
    >>> pulse = lambda x: np.exp(-(x**2) / 0.01)
    >>> # the pulse split in two halves
    >>> dalembert([0.0, 1.0], 1.0, pulse, 1.0).round(4)
    array([0. , 0.5])
    >>> # sin(ct)/c
    >>> round(
    ...     float(dalembert(0.0, 0.5, np.zeros_like, 1.0, initial_rate=np.cos)), 6
    ... )
    0.479426
    """
    if speed <= 0:
        raise ValueError(f"speed must be positive; got {speed!r}")
    x, t = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
    u = 0.5 * (initial(x - speed * t) + initial(x + speed * t))
    if initial_rate is not None:
        nodes, weights = _GAUSS_LEGENDRE
        half = speed * t
        values = initial_rate(x[..., None] + half[..., None] * nodes)
        u = u + (values @ weights) * half / (2 * speed)
    return _out(u)


# ---------------------------------------------------------------------------------------------
# Pressureless flow and caustics
# ---------------------------------------------------------------------------------------------
@dataclass
class LagrangianFlow:
    """A pressureless (ballistic) flow followed along its fluid elements.

    Attributes
    ----------
    position : float or numpy.ndarray
        The Eulerian position x = q + v0(q) t of each element.
    velocity : float or numpy.ndarray
        Its velocity v0(q), constant in time.
    density : float or numpy.ndarray
        The density at x, ρ0(q)/|1 + v0'(q) t|; ``inf`` at a caustic.
    """

    position: object
    velocity: object
    density: object


def _derivative(function, q):
    """A centered finite-difference derivative (relative accuracy about 1e-10)."""
    h = 1e-5 * np.maximum(1.0, np.abs(q))
    return (function(q + h) - function(q - h)) / (2 * h)


def pressureless(q, t, velocity, density=None, velocity_derivative=None):
    """Follow a 1-D pressureless flow (the Zel'dovich approximation) in Lagrangian coordinates.

    Without pressure every fluid element keeps its initial velocity, x(q, t) = q + v0(q) t, and
    mass conservation ρ dx = ρ0 dq gives ρ = ρ0(q)/|∂x/∂q| = ρ0(q)/|1 + v0'(q) t|. Where the
    velocity decreases, elements catch up with each other: the density becomes infinite (a
    caustic) at :func:`caustic_time`, after which the map is multivalued (shell crossing) and
    the density here is that of each stream.

    Parameters
    ----------
    q : float or array_like
        The initial (Lagrangian) positions of fluid elements.
    t : float or array_like
        Times, broadcast against ``q``.
    velocity : callable
        The initial velocity v0(q), a vectorized function.
    density : callable, optional
        The initial density ρ0(q), a vectorized function. Default: ``None`` (uniform, 1).
    velocity_derivative : callable, optional
        v0'(q). Default: ``None`` (by finite differences).

    Returns
    -------
    LagrangianFlow
        The positions, velocities and densities of the elements.

    See Also
    --------
    pressureless_eulerian : The same flow as a function of the position x.
    caustic_time : The time of the first shell crossing.

    References
    ----------
    Ya. B. Zel'dovich, "Gravitational instability: an approximate theory for large density
    perturbations", Astron. Astrophys. 5, 84 (1970).

    Examples
    --------
    A converging sine velocity, v0 = −0.5 sin(q), with its caustic at t = 2:

    >>> flow = pressureless([0.0, np.pi / 2], 1.0, lambda q: -0.5 * np.sin(q))
    >>> flow.position.round(6), flow.density.round(6)
    (array([0.      , 1.070796]), array([2., 1.]))
    """
    q, t = np.broadcast_arrays(np.asarray(q, dtype=float), np.asarray(t, dtype=float))
    v = velocity(q)
    dv = velocity_derivative(q) if velocity_derivative is not None else _derivative(velocity, q)
    rho0 = density(q) if density is not None else 1.0
    with np.errstate(divide="ignore"):
        rho = rho0 / np.abs(1 + dv * t)
    return LagrangianFlow(_out(q + v * t), _out(v + 0 * q), _out(rho + 0 * q))


def pressureless_eulerian(x, t, velocity, density=None, velocity_derivative=None):
    """Compute a 1-D pressureless flow as a function of the position, before shell crossing.

    Inverts x = q + v0(q) t for the Lagrangian position q (by Newton's method safeguarded by
    bisection, to machine precision) and returns the flow there. The inverse is unique only
    before the first caustic (:func:`caustic_time`); later, one of the streams is returned.

    Parameters
    ----------
    x : float or array_like
        Positions.
    t : float or array_like
        Times, broadcast against ``x``.
    velocity : callable
        The initial velocity v0(q), a vectorized function.
    density : callable, optional
        The initial density ρ0(q), a vectorized function. Default: ``None`` (uniform, 1).
    velocity_derivative : callable, optional
        v0'(q). Default: ``None`` (by finite differences).

    Returns
    -------
    LagrangianFlow
        ``position`` is x itself; the velocity and density at x.

    See Also
    --------
    pressureless : The flow along the fluid elements.

    References
    ----------
    Ya. B. Zel'dovich, "Gravitational instability: an approximate theory for large density
    perturbations", Astron. Astrophys. 5, 84 (1970).

    Examples
    --------
    >>> flow = pressureless_eulerian(
    ...     [-1.070796, 1.070796], 1.0, lambda q: -0.5 * np.sin(q)
    ... )
    >>> flow.density.round(5), flow.velocity.round(5)
    (array([1., 1.]), array([ 0.5, -0.5]))
    """
    x, t = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(t, dtype=float))
    x, t = x.copy(), t.copy()

    def dv(q):
        return velocity_derivative(q) if velocity_derivative is not None else _derivative(velocity, q)

    def g(q):
        return q + velocity(q) * t - x

    # bracket the root: expand from the positions until g changes sign
    step = np.abs(velocity(x) * t) + 1e-12 + 1e-12 * np.abs(x)
    low, high = x - step, x + step
    for _ in range(200):
        bad_low, bad_high = g(low) > 0, g(high) < 0
        if not (bad_low.any() or bad_high.any()):
            break
        step = 2 * step
        low, high = np.where(bad_low, low - step, low), np.where(bad_high, high + step, high)
    q = 0.5 * (low + high)
    for _ in range(200):
        value = g(q)
        low, high = np.where(value < 0, q, low), np.where(value < 0, high, q)
        with np.errstate(divide="ignore", invalid="ignore"):
            new = q - value / (1 + dv(q) * t)
        new = np.where((new > low) & (new < high), new, 0.5 * (low + high))
        done = np.abs(new - q) <= 4e-16 * np.maximum(1.0, np.abs(q))
        q = new
        if done.all():
            break
    flow = pressureless(q, t, velocity, density, velocity_derivative)
    return LagrangianFlow(_out(x), flow.velocity, flow.density)


def caustic_time(velocity, q):
    """Compute the time of the first caustic (shell crossing) of a pressureless flow.

    Elements with v0'(q) < 0 meet at t = −1/v0'(q), so the first caustic forms at
    t_c = −1/min v0'(q), at the steepest decrease of the initial velocity.

    Parameters
    ----------
    velocity : callable
        The initial velocity v0(q), a vectorized function.
    q : array_like
        The positions where v0' is sampled (the minimum is taken over these, so resolve it).

    Returns
    -------
    float
        t_c; ``inf`` if the velocity nowhere decreases.

    See Also
    --------
    pressureless : The flow up to (and past) the caustic.

    References
    ----------
    Ya. B. Zel'dovich, "Gravitational instability: an approximate theory for large density
    perturbations", Astron. Astrophys. 5, 84 (1970).

    Examples
    --------
    >>> q = np.linspace(-np.pi, np.pi, 1001)
    >>> round(caustic_time(lambda q: -0.5 * np.sin(q), q), 6)
    2.0
    """
    slope = float(np.min(_derivative(velocity, np.asarray(q, dtype=float))))
    return -1 / slope if slope < 0 else np.inf


# ---------------------------------------------------------------------------------------------
# Decaying modes
# ---------------------------------------------------------------------------------------------


def _wavenumber_squared(wavenumber):
    k = np.asarray(wavenumber, dtype=float)
    return float(np.sum(k**2)) if k.ndim == 1 else k**2


# ---------------------------------------------------------------------------------------------
# Incompressible flows
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Electromagnetic cavity modes
# ---------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------
# Manufactured solutions
# ---------------------------------------------------------------------------------------------
