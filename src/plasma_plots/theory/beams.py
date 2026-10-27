"""Charged-particle beams: perveance, ion optics, beam envelopes and beam–gas processes, in SI units.

Voltages are in V, currents in A, lengths in m, angles in rad (unless a function says otherwise),
densities in m⁻³, cross sections in m² and speeds in m/s. Particles have the mass ``mass`` (kg) and the
charge ``charge`` (C); the defaults are electrons, as in :mod:`~plasma_plots.theory.parameters`, which
this module reuses for the constants and the plasma frequency. Pass ``mass=proton_mass`` or
``mass=mass_number * proton_mass`` for ions.

Conventions
-----------
* **Perveance**: ``perveance`` returns either the dimensional P = I/V^(3/2) in A/V^(3/2), or the
  dimensionless generalized perveance K = q I / (2π ε₀ m γ³ v³) = 2 I / (I₀ β³ γ³) with the
  characteristic current I₀ = 4π ε₀ m c³ / q, the K that enters the envelope equation.
* **Envelopes**: radii are rms radii x_rms of a round beam and emittances are unnormalized rms
  emittances ε in m·rad. A Kapchinskij–Vladimirskij (KV) beam has the edge radius a = 2 x_rms and the
  edge emittance 4ε, so the equations also hold for a KV edge radius with ε → 4ε and K/4 → K.

References
----------
M. Reiser, Theory and Design of Charged Particle Beams, 2nd ed. (Wiley, 2008).

S. Humphries, Charged Particle Beams (Wiley, 1990).

I. G. Brown (ed.), The Physics and Technology of Ion Sources, 2nd ed. (Wiley, 2004).

Examples
--------
The Child–Langmuir limit of a 5 mm radius, 4 mm gap proton extraction aperture at 30 kV, and the
generalized perveance of a 1 A, 30 keV proton beam:

>>> print(
...     f"{child_langmuir_aperture(30e3, 5e-3, 4e-3, mass=proton_mass):.4g} A"
... )
1.389 A
>>> print(f"{perveance(1.0, 30e3, mass=proton_mass, kind='generalized'):.4g}")
0.125
"""

from __future__ import annotations

import numpy as np

from plasma_plots.theory.parameters import (
    electron_mass,
    elementary_charge,
    plasma_frequency,
    proton_mass,  # noqa: F401  (used in the doctests)
    speed_of_light,
    vacuum_permittivity,
)

_PERVEANCE_KINDS = ("A/V^1.5", "generalized")


def _out(value):
    return np.asarray(value, dtype=float)[()]


def beam_speed(
    voltage, mass=electron_mass, charge=elementary_charge, relativistic=False
):
    """Compute the speed v and the Lorentz factor γ of a particle accelerated through a voltage.

    Parameters
    ----------
    voltage : float or array_like
        Accelerating voltage V in V, so the kinetic energy is q V.
    mass, charge : float or array_like, optional
        Particle mass (kg) and charge (C). Default: electron.
    relativistic : bool, optional
        If True, γ = 1 + q V / (m c²) and v = c √(1 − 1/γ²); otherwise v = √(2 q V / m) and γ = 1.

    Returns
    -------
    speed, gamma : float or numpy.ndarray

    References
    ----------
    Reiser, Theory and Design of Charged Particle Beams, ch. 2 (energy and velocity).

    Examples
    --------
    >>> v, gamma = beam_speed(511e3, relativistic=True)
    >>> print(f"{v / speed_of_light:.4f} {gamma:.4f}")
    0.8660 2.0000
    """
    voltage, mass, charge = (
        np.asarray(x, dtype=float) for x in (voltage, mass, charge)
    )
    if relativistic:
        gamma = 1.0 + charge * voltage / (mass * speed_of_light**2)
        speed = speed_of_light * np.sqrt(1.0 - 1.0 / gamma**2)
    else:
        gamma = np.ones(np.broadcast(voltage, mass, charge).shape)
        speed = np.sqrt(2.0 * charge * voltage / mass)
    return _out(speed), _out(gamma)


def perveance(
    current,
    voltage,
    mass=electron_mass,
    charge=elementary_charge,
    kind="A/V^1.5",
    relativistic=False,
):
    """Compute the perveance of a beam.

    ``kind="A/V^1.5"`` gives P = I / V^(3/2). ``kind="generalized"`` gives the dimensionless
    K = q I / (2π ε₀ m γ³ v³), which is 2 I / (I₀ β³ γ³) with I₀ = 4π ε₀ m c³ / q and is the space-charge
    strength in the envelope equation (:func:`kv_envelope`).

    Parameters
    ----------
    current : float or array_like
        Beam current I in A.
    voltage : float or array_like
        Accelerating voltage V in V.
    mass, charge : float or array_like, optional
        Particle mass (kg) and charge (C). Default: electron.
    kind : {"A/V^1.5", "generalized"}, optional
        Which perveance to return.
    relativistic : bool, optional
        Use the relativistic v and γ for the generalized perveance (see :func:`beam_speed`).

    Returns
    -------
    float or numpy.ndarray
        P in A/V^(3/2), or the dimensionless K.

    References
    ----------
    Reiser, Theory and Design of Charged Particle Beams, 2nd ed., secs. 4.1 and 4.2 (eqs. for the
    perveance and the generalized perveance K).

    Examples
    --------
    >>> print(f"{perveance(1.0, 30e3, mass=proton_mass):.4g} A/V^1.5")
    1.925e-07 A/V^1.5
    >>> print(f"{perveance(1.0, 30e3, mass=proton_mass, kind='generalized'):.4g}")
    0.125
    """
    if kind not in _PERVEANCE_KINDS:
        raise ValueError(f"kind must be one of {_PERVEANCE_KINDS}, got {kind!r}")
    current, voltage = (
        np.asarray(current, dtype=float),
        np.asarray(voltage, dtype=float),
    )
    if kind == "A/V^1.5":
        return _out(current / voltage**1.5)
    speed, gamma = beam_speed(voltage, mass, charge, relativistic)
    return _out(
        charge
        * current
        / (2 * np.pi * vacuum_permittivity * mass * gamma**3 * speed**3)
    )


def child_langmuir_aperture(
    voltage, radius, gap, mass=electron_mass, charge=elementary_charge
):
    """Compute the space-charge-limited current through a circular aperture, I = π r² J_CL.

    J_CL = (4 ε₀ / 9) √(2 q / m) V^(3/2) / d² is the planar Child–Langmuir current density across the
    gap d, taken uniform over the aperture. Edge effects (the Kirstein–Coupland corrections for
    large r/d) are not included, so this is the leading-order limit for r ≲ d.

    Parameters
    ----------
    voltage : float or array_like
        Extraction voltage V in V.
    radius : float or array_like
        Aperture radius r in m.
    gap : float or array_like
        Gap d between the emitting surface (plasma meniscus) and the electrode, in m.
    mass, charge : float or array_like, optional
        Particle mass (kg) and charge (C). Default: electron.

    Returns
    -------
    float or numpy.ndarray
        The current in A.

    References
    ----------
    C. D. Child, Phys. Rev. (Series I) 32, 492 (1911); I. Langmuir, Phys. Rev. 2, 450 (1913).
    Brown (ed.), The Physics and Technology of Ion Sources, 2nd ed., ch. 2.

    Examples
    --------
    >>> print(
    ...     f"{child_langmuir_aperture(30e3, 5e-3, 4e-3, mass=proton_mass):.4g} A"
    ... )
    1.389 A
    """
    voltage, radius, gap = (np.asarray(x, dtype=float) for x in (voltage, radius, gap))
    j_cl = (
        4 * vacuum_permittivity / 9 * np.sqrt(2 * charge / mass) * voltage**1.5 / gap**2
    )
    return _out(np.pi * radius**2 * j_cl)


def optimal_perveance(aspect_ratio, mass=electron_mass, charge=elementary_charge):
    """Compute the perveance P = I/V^(3/2) of the optimum (minimum-divergence) extraction, in A/V^1.5.

    The optimum is the Child–Langmuir perveance of the aperture, P_opt = (4π ε₀ / 9) √(2 q / m) (r/d)²
    with the aspect ratio r/d of aperture radius over gap. This is the planar space-charge-limited
    scaling, used here as the leading-order estimate of the perveance of minimum divergence; it
    ignores the curvature of the plasma meniscus and the fringe fields, so real optima deviate from
    it. It equals ``child_langmuir_aperture(V, r, d) / V**1.5``; this function takes only the geometry.

    Parameters
    ----------
    aspect_ratio : float or array_like
        Aperture radius over gap, r/d.
    mass, charge : float or array_like, optional
        Particle mass (kg) and charge (C). Default: electron.

    Returns
    -------
    float or numpy.ndarray
        P_opt in A/V^(3/2).

    References
    ----------
    Brown (ed.), The Physics and Technology of Ion Sources, 2nd ed., ch. 2. The existence of a
    perveance of minimum divergence for a three-electrode aperture is studied in J. R. Coupland,
    T. S. Green, D. P. Hammond and A. C. Riviere, "A Study of the Ion Beam Intensity and Divergence
    Obtained from a Single Aperture Three Electrode Extraction System", Rev. Sci. Instrum. 44, 1258
    (1973); that paper's own optimum-perveance formula was not checked against this one.

    Examples
    --------
    Electrons and protons at r/d = 1 (the ratio is √(m_p/m_e) = 42.8):

    >>> print(
    ...     f"{optimal_perveance(1.0):.4g} {optimal_perveance(1.0, mass=proton_mass):.4g}"
    ... )
    7.332e-06 1.711e-07
    """
    aspect_ratio = np.asarray(aspect_ratio, dtype=float)
    return _out(
        4
        * np.pi
        * vacuum_permittivity
        / 9
        * np.sqrt(2 * charge / mass)
        * aspect_ratio**2
    )


def kv_envelope(z, radius0, slope0, emittance):
    """Compute the rms envelope of a beam in a drift without space charge, in closed form.

    Solves x'' = ε²/x³ (the Kapchinskij–Vladimirskij envelope equation for K = 0 and no focusing):
    x(z)² = x₀² + 2 x₀ x₀' z + (x₀'² + ε²/x₀²) z². With a waist (x₀' = 0) this is the hyperbola
    x² = x₀² + (ε z / x₀)². With space charge or focusing use :func:`kv_envelope_ode`.

    Parameters
    ----------
    z : float or array_like
        Distance along the drift in m.
    radius0 : float or array_like
        rms radius x₀ at z = 0, in m.
    slope0 : float or array_like
        Envelope slope x₀' at z = 0 (rad).
    emittance : float or array_like
        Unnormalized rms emittance ε in m·rad.

    Returns
    -------
    float or numpy.ndarray
        The rms radius x(z) in m.

    References
    ----------
    F. J. Sacherer, IEEE Trans. Nucl. Sci. 18, 1105 (1971); Reiser, Theory and Design of Charged
    Particle Beams, 2nd ed., sec. 4.2 (envelope equation) and the drift solution of
    Humphries, Charged Particle Beams, ch. 5.

    Examples
    --------
    A 1 mm waist with ε = 1 mm·mrad, 10 m downstream:

    >>> print(f"{kv_envelope(10.0, 1e-3, 0.0, 1e-6) * 1e3:.4g} mm")
    10.05 mm
    """
    z, x0, s0, eps = (
        np.asarray(v, dtype=float) for v in (z, radius0, slope0, emittance)
    )
    return _out(np.sqrt(x0**2 + 2 * x0 * s0 * z + (s0**2 + eps**2 / x0**2) * z**2))


def kv_envelope_ode(
    z, radius0, slope0, emittance, perveance=0.0, focusing=None, steps_per_interval=50
):
    """Integrate the KV rms envelope equation with space charge and focusing, using RK4.

    Solves x'' + κ(z) x = ε²/x³ + K/(4 x) for a round beam with the generalized perveance K
    (:func:`perveance`, ``kind="generalized"``) and the focusing strength κ(z) in m⁻². The K/(4x) term is
    the space-charge force of a uniform-density beam (KV edge radius a = 2x obeys a'' = K/a). The
    integrator is a hand-written fixed-step fourth-order Runge–Kutta.

    Parameters
    ----------
    z : array_like
        Increasing 1-D positions in m; the beam starts at ``z[0]`` with ``radius0`` and ``slope0``.
    radius0, slope0 : float or array_like
        rms radius (m) and its slope at ``z[0]``; they broadcast with the other parameters.
    emittance : float or array_like
        Unnormalized rms emittance ε in m·rad.
    perveance : float or array_like, optional
        Generalized perveance K (dimensionless). Default: 0.
    focusing : callable, optional
        ``focusing(z)`` returns κ(z) in m⁻² (a scalar or something that broadcasts with the radius).
        Default: a drift (κ = 0).
    steps_per_interval : int, optional
        RK4 steps between consecutive entries of ``z``.

    Returns
    -------
    radius, slope : numpy.ndarray
        x and x' at each ``z``, shape ``(len(z), *broadcast shape)``.

    References
    ----------
    Reiser, Theory and Design of Charged Particle Beams, 2nd ed., sec. 4.2; M. Lund and B. Bukh,
    Phys. Rev. ST Accel. Beams 7, 024801 (2004), eq. for x'' with the perveance Q.

    Examples
    --------
    Without space charge this reproduces :func:`kv_envelope`:

    >>> x, _ = kv_envelope_ode([0.0, 10.0], 1e-3, 0.0, 1e-6)
    >>> print(f"{x[-1] * 1e3:.4g} mm")
    10.05 mm
    """
    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or z.size < 1 or np.any(np.diff(z) <= 0):
        raise ValueError("z must be a 1-D array of increasing positions")
    x, s, eps, k_perv = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (radius0, slope0, emittance, perveance))
    )
    x, s = x.copy(), s.copy()

    def accel(zz, xx):
        kappa = 0.0 if focusing is None else focusing(zz)
        return eps**2 / xx**3 + k_perv / (4 * xx) - kappa * xx

    radius, slope = [x.copy()], [s.copy()]
    for z0, z1 in zip(z[:-1], z[1:]):
        h = (z1 - z0) / steps_per_interval
        for i in range(steps_per_interval):
            zz = z0 + i * h
            k1x, k1s = s, accel(zz, x)
            k2x, k2s = s + h / 2 * k1s, accel(zz + h / 2, x + h / 2 * k1x)
            k3x, k3s = s + h / 2 * k2s, accel(zz + h / 2, x + h / 2 * k2x)
            k4x, k4s = s + h * k3s, accel(zz + h, x + h * k3x)
            x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
            s = s + h / 6 * (k1s + 2 * k2s + 2 * k3s + k4s)
        radius.append(x.copy())
        slope.append(s.copy())
    return np.array(radius), np.array(slope)


def beam_divergence_gaussian(ion_temperature, beam_energy, definition="1/e"):
    """Compute the divergence of a beam with a Maxwellian transverse velocity spread.

    The transverse velocity in each plane has the standard deviation √(T/m) and the axial speed is
    √(2E/m), so the rms divergence is θ_rms = √(T / (2E)) and the half angle at which the angular
    distribution has fallen to 1/e is θ_1/e = √2 θ_rms = √(T/E) (small angles).

    Parameters
    ----------
    ion_temperature : float or array_like
        Transverse temperature T of the extracted particles, in eV.
    beam_energy : float or array_like
        Beam kinetic energy E, in eV.
    definition : {"1/e", "rms"}, optional
        Which divergence to return.

    Returns
    -------
    float or numpy.ndarray
        The divergence in rad.

    References
    ----------
    Humphries, Charged Particle Beams, ch. 5 (emittance from a thermal source);
    Brown (ed.), The Physics and Technology of Ion Sources, 2nd ed., ch. 2.

    Examples
    --------
    A 1 eV ion temperature at 30 keV:

    >>> print(f"{beam_divergence_gaussian(1.0, 30e3) * 1e3:.4g} mrad")
    5.774 mrad
    """
    if definition not in ("1/e", "rms"):
        raise ValueError(f"definition must be '1/e' or 'rms', got {definition!r}")
    ratio = np.asarray(ion_temperature, dtype=float) / np.asarray(
        beam_energy, dtype=float
    )
    return _out(np.sqrt(ratio if definition == "1/e" else ratio / 2))


def pierce_angle(degrees=True):
    """Return the Pierce electrode angle, 67.5° (3π/8) from the beam axis.

    For an infinite planar beam edge, the Pierce electrode holds the equipotential of the
    Child–Langmuir flow along a plane inclined by 67.5° to the axis (22.5° to the emitting surface),
    which makes the beam edge straight and parallel.

    Parameters
    ----------
    degrees : bool, optional
        Return degrees (default) or radians.

    Returns
    -------
    float
        The angle between the focusing electrode and the beam axis.

    References
    ----------
    J. R. Pierce, J. Appl. Phys. 11, 548 (1940); Pierce, Theory and Design of Electron Beams (Van
    Nostrand, 1954); Humphries, Charged Particle Beams, ch. 5.

    Examples
    --------
    >>> pierce_angle()
    67.5
    """
    return 67.5 if degrees else 3 * np.pi / 8


def aperture_lens_focal_length(voltage, field_before, field_after):
    """Compute the focal length of a thin aperture lens, f = 4V / (E₂ − E₁) (Davisson–Calbick).

    Parameters
    ----------
    voltage : float or array_like
        Potential of the aperture relative to the cathode, i.e. the kinetic energy per charge at the
        aperture, in V.
    field_before : float or array_like
        Axial field E₁ upstream of the aperture, in V/m, positive when it accelerates the particles
        along their direction of motion (independent of the sign of the charge).
    field_after : float or array_like
        The same downstream, E₂.

    Returns
    -------
    float or numpy.ndarray
        The focal length in m: positive for a converging lens (E₂ > E₁, e.g. the entrance of an
        accelerating gap), negative for a diverging one, infinite if E₁ = E₂.

    References
    ----------
    C. J. Davisson and C. J. Calbick, Phys. Rev. 38, 585 (1931); Humphries, Charged Particle Beams,
    ch. 5 (aperture lens); Reiser, Theory and Design of Charged Particle Beams, 2nd ed., ch. 3.

    Examples
    --------
    A 30 kV aperture with 0 on the plasma side and 10 kV/cm after it:

    >>> print(f"{aperture_lens_focal_length(30e3, 0.0, 1e6):.4g} m")
    0.12 m
    """
    voltage, e1, e2 = (
        np.asarray(x, dtype=float) for x in (voltage, field_before, field_after)
    )
    with np.errstate(divide="ignore"):
        return _out(4 * voltage / (e2 - e1))


def beam_plasma_frequency(
    density,
    mass=electron_mass,
    charge=elementary_charge,
    gamma=1.0,
    direction="longitudinal",
):
    """Compute the beam plasma frequency of a beam of density n in its rest frame, seen from the lab.

    ω_pb = √(n q² / (ε₀ γ³ m)) for longitudinal oscillations and √(n q² / (ε₀ γ m)) for transverse
    ones, with n the lab-frame density (the non-relativistic limit is
    :func:`~plasma_plots.theory.parameters.plasma_frequency`).

    Parameters
    ----------
    density : float or array_like
        Beam density n in the lab frame, in m⁻³ (``current / (q v π r²)`` for a uniform round beam).
    mass, charge : float or array_like, optional
        Particle mass (kg) and charge (C). Default: electron.
    gamma : float or array_like, optional
        Lorentz factor of the beam. Default: 1.
    direction : {"longitudinal", "transverse"}, optional

    Returns
    -------
    float or numpy.ndarray
        ω_pb in rad/s.

    References
    ----------
    Reiser, Theory and Design of Charged Particle Beams, 2nd ed., ch. 4 (space-charge
    waves); Humphries, Charged Particle Beams, ch. 7.

    Examples
    --------
    A 1e16 m⁻³ proton beam:

    >>> print(f"{beam_plasma_frequency(1e16, mass=proton_mass):.4g} rad/s")
    1.317e+08 rad/s
    """
    powers = {"longitudinal": 3, "transverse": 1}
    if direction not in powers:
        raise ValueError(f"direction must be one of {tuple(powers)}, got {direction!r}")
    gamma = np.asarray(gamma, dtype=float)
    return _out(
        plasma_frequency(density, mass, charge) / gamma ** (powers[direction] / 2)
    )


def _along(positions, ndim, axis):
    """The shape of ``positions`` placed on ``axis`` of an ``ndim``-dimensional array."""
    shape = [1] * max(ndim, 1)
    shape[axis] = positions.size
    return tuple(shape)


def neutral_beam_attenuation(
    density, cross_section, length=None, positions=None, axis=-1
):
    """Compute the surviving fraction of a beam, exp(−∫ n σ dl).

    Use either a path ``length`` for a constant n σ, or ``positions`` (sample points along the
    path) with arrays of the density and/or cross section there, integrated with the trapezoidal
    rule along ``axis``. Particles are lost at the rate n σ per unit length by ionisation,
    charge exchange or stripping, so this serves both for a neutral beam in a plasma or gas
    target and for a negative-ion beam stripped on gas.

    Parameters
    ----------
    density : float or array_like
        Target density n in m⁻³.
    cross_section : float or array_like
        Total loss cross section σ in m² (a constant, or an array e.g. from σ(E(l))).
    length : float or array_like, optional
        Path length in m, for constant n σ.
    positions : array_like, optional
        Increasing positions l along ``axis`` of the arrays, in m; given instead of ``length``.
    axis : int, optional
        The axis of ``density``/``cross_section`` along which ``positions`` runs.

    Returns
    -------
    float or numpy.ndarray
        The fraction of the beam that is neither ionised nor stripped.

    References
    ----------
    Brown (ed.), The Physics and Technology of Ion Sources, 2nd ed., ch. on negative ions and neutral
    beams; the beam-attenuation law of J. Wesson, Tokamaks, 4th ed. (OUP, 2011).

    Examples
    --------
    1e19 m⁻³ at σ = 1e-20 m² over 5 m:

    >>> print(f"{neutral_beam_attenuation(1e19, 1e-20, length=5.0):.4f}")
    0.6065
    """
    if (length is None) == (positions is None):
        raise ValueError("pass exactly one of length and positions")
    rate = np.asarray(density, dtype=float) * np.asarray(cross_section, dtype=float)
    if length is not None:
        return _out(np.exp(-rate * np.asarray(length, dtype=float)))
    positions = np.asarray(positions, dtype=float)
    rate = np.broadcast_to(
        rate, np.broadcast_shapes(rate.shape, _along(positions, rate.ndim, axis))
    )
    rate = np.moveaxis(rate, axis, -1)
    optical_depth = np.sum(
        0.5 * (rate[..., 1:] + rate[..., :-1]) * np.diff(positions), axis=-1
    )
    return _out(np.exp(-optical_depth))


def stripping_fraction(density, cross_section, length=None, positions=None, axis=-1):
    """Compute the fraction of a beam lost (stripped or ionised) on a target, 1 − exp(−∫ n σ dl).

    The complement of :func:`neutral_beam_attenuation` and takes the same arguments: the fraction
    of a negative-ion beam that is neutralised in a gas cell, or of an atomic beam that is ionised.

    Parameters
    ----------
    density, cross_section, length, positions, axis
        See :func:`neutral_beam_attenuation`.

    Returns
    -------
    float or numpy.ndarray

    References
    ----------
    Brown (ed.), The Physics and Technology of Ion Sources, 2nd ed.; Wesson, Tokamaks, 4th ed.

    Examples
    --------
    >>> print(f"{stripping_fraction(1e19, 1e-20, length=5.0):.4f}")
    0.3935
    """
    return _out(
        1.0 - neutral_beam_attenuation(density, cross_section, length, positions, axis)
    )


def space_charge_compensation_time(gas_density, cross_section, beam_speed):
    """Compute the time to space-charge compensate a beam by ionising the background gas.

    The beam ionises the gas at the rate n_b n_g σ v_b per unit volume, and the produced slow ions
    (or electrons) are trapped by the beam potential, so the compensating density reaches the beam
    density n_b after τ = 1 / (n_g σ v_b). This is the build-up time in the absence of losses.

    Parameters
    ----------
    gas_density : float or array_like
        Background gas density n_g in m⁻³ (p / k_B T).
    cross_section : float or array_like
        Beam-impact ionisation cross section σ in m².
    beam_speed : float or array_like
        Beam speed v_b in m/s (:func:`beam_speed`).

    Returns
    -------
    float or numpy.ndarray
        τ in s.

    References
    ----------
    The estimate τ = 1/(n_g σ v_b) follows from the ionisation rate above. For the physics of
    space-charge compensation of ion beams by the gas-ionisation plasma see the review of M. D.
    Gabovich, "Ion-beam plasma and the propagation of intense compensated ion beams", Sov. Phys.
    Usp. 20, 134 (1977) (only its abstract was checked); Reiser, Theory and Design of Charged
    Particle Beams, 2nd ed., ch. 4 (neutralisation).

    Examples
    --------
    30 keV protons in a gas of 2.4e17 m⁻³ (about 1e-3 Pa) with σ = 1e-20 m²:

    >>> v, _ = beam_speed(30e3, mass=proton_mass)
    >>> print(f"{space_charge_compensation_time(2.4e17, 1e-20, v) * 1e6:.4g} us")
    173.8 us
    """
    return _out(
        1.0
        / (
            np.asarray(gas_density, dtype=float)
            * np.asarray(cross_section, dtype=float)
            * np.asarray(beam_speed, dtype=float)
        )
    )


def emittance_growth_space_charge(
    field_energy_initial, tune_depression, field_energy_final=0.0
):
    """Compute the rms emittance growth ε_f/ε_i of a non-uniform beam from its field energy.

    For a round, axisymmetric beam in continuous focusing, the nonlinear space-charge force changes
    the rms emittance according to Wangler's theorem,
    d(ε²)/ds = −32π ε₀ K ⟨x²⟩ d[(W − W_u)/λ²]/ds (the edge emittance, 16 times the squared rms
    emittance), where W is the field energy per unit length of the beam, W_u that of the rms-equivalent
    uniform beam and λ the line charge density. If the rms radius stays constant and the beam starts
    matched, with the tune depression μ = σ/σ₀ (so K x_rms² = 4 k₀² x_rms⁴ (1 − μ²) and
    ε² = k₀² x_rms⁴ μ²), this integrates to

    ε_f²/ε_i² = 1 + ½ (1/μ² − 1) (u_i − u_f),   u = U/w₀ = 16π ε₀ (W − W_u)/λ²,

    with the dimensionless nonlinear field energy factor u (w₀ = λ²/(16π ε₀)) of the initial and
    final charge distributions. It is zero for a uniform (KV) distribution, so a beam that relaxes
    to a uniform one, u_f = 0, has the largest growth. The ½ (1/μ² − 1) form with u_f = 0 is
    Eq. (1) of Kikuchi and Horioka, who quote it from Reiser; the u_f ≠ 0 generalisation and the
    reduction to the tune depression were derived from the Lund slides' Wangler theorem as above.

    Parameters
    ----------
    field_energy_initial : float or array_like
        u_i = U/w₀ of the initial distribution. For a Gaussian it is 2γ_E − 1 = 0.1544
        (γ_E is Euler's constant), for a KV beam 0.
    tune_depression : float or array_like
        μ = σ/σ₀ of the initial matched beam, 0 < μ ≤ 1.
    field_energy_final : float or array_like, optional
        u_f of the final distribution. Default: 0 (relaxation to a uniform beam).

    Returns
    -------
    float or numpy.ndarray
        The ratio ε_f/ε_i of the rms emittances. Values below 1 occur for u_f > u_i.

    Raises
    ------
    ValueError
        If ε_f²/ε_i² would be negative, which means the assumptions (constant rms radius) fail.

    References
    ----------
    T. Kikuchi and K. Horioka, Proc. PAC09, FR5PFP038 (2009), Eq. (1), and the Wangler theorem as
    derived in S. M. Lund, USPAS lecture notes on transverse kinetic stability, 2017, slides 61-73
    (cites T. P. Wangler et al., IEEE Trans. Nucl. Sci. 32, 2196 (1985), and Reiser). Both were read;
    Reiser's book was not.

    Examples
    --------
    A Gaussian beam with tune depression 0.2 that relaxes to a uniform beam:

    >>> print(f"{emittance_growth_space_charge(2 * np.euler_gamma - 1, 0.2):.4f}")
    1.6891
    """
    u_i, mu, u_f = (
        np.asarray(x, dtype=float)
        for x in (field_energy_initial, tune_depression, field_energy_final)
    )
    if np.any(mu <= 0) or np.any(mu > 1):
        raise ValueError("tune_depression must be in (0, 1]")
    ratio_sq = 1.0 + 0.5 * (1.0 / mu**2 - 1.0) * (u_i - u_f)
    if np.any(ratio_sq < 0):
        raise ValueError(
            "the field-energy change gives a negative emittance; the constant-radius assumption fails"
        )
    return _out(np.sqrt(ratio_sq))
