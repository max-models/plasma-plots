"""Sheaths, space-charge-limited flow, probes and low-pressure discharges, in SI units.

Densities are in m⁻³, temperatures in eV (k_B T / e), voltages in V, lengths in m and currents in
A, as in :mod:`~plasma_plots.theory.parameters`. Ions have the mass ``mass_number × m_p`` and the
charge ``charge_number × e``; the functions that apply to any particle (space-charge-limited
flow, probes) take ``mass`` in kg and ``charge`` in C instead, with the electron as the default.

Contents
--------
* **Sheaths**: :func:`bohm_speed`, :func:`bohm_current_density`, :func:`floating_potential`,
  :func:`sheath_width`.
* **Space-charge-limited flow**: :func:`child_langmuir` (planar, with an optional injection
  speed) and :func:`langmuir_blodgett` (coaxial cylinders and concentric spheres).
* **Probes**: :func:`oml_current` and :func:`ion_saturation_current`.
* **Transport and breakdown**: :func:`ambipolar_diffusion`, :func:`bohm_diffusion`,
  :func:`paschen_breakdown` and :func:`mean_free_path`.

Conventions
-----------
* **Potentials** are magnitudes unless stated: ``voltage`` is the potential difference that
  accelerates (or repels) the particles, never negative in the result of a sheath width.
  :func:`floating_potential` is the exception and returns the (negative) wall potential relative
  to the plasma.
* **Sheath-edge density** is the bulk density times exp(−1/2) in the Bohm current: the
  presheath drop of Te/2 of a collisionless plasma.

References
----------
M. A. Lieberman and A. J. Lichtenberg, Principles of Plasma Discharges and Materials
Processing, 2nd ed. (Wiley, 2005).

F. F. Chen, Introduction to Plasma Physics and Controlled Fusion, 3rd ed. (Springer, 2016).

I. Langmuir and K. B. Blodgett, Phys. Rev. 24, 49 (1924).

Examples
--------
The Bohm speed and ion saturation current density of a 1e18 m⁻³, 5 eV hydrogen plasma:

>>> print(
...     f"{bohm_speed(5.0):.4g} m/s, {bohm_current_density(1e18, 5.0):.4g} A/m²"
... )
2.188e+04 m/s, 2127 A/m²
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from plasma_plots.theory.parameters import (
    boltzmann_constant,
    debye_length,
    electron_mass,
    elementary_charge,
    ev_to_joule,
    proton_mass,
    sound_speed,
    vacuum_permittivity,
)

_e = elementary_charge
_SHEATH_EDGE = math.exp(-0.5)

#: Constants (A in 1/(Pa m), B in V/(Pa m)) of the Paschen law for some gases, converted from
#: the usual 1/(cm Torr) and V/(cm Torr). These are fits of the first Townsend coefficient,
#: α/p = A exp(−B p/E), valid only over a limited range of E/p (V/(cm Torr)), with the
#: values of the standard compilation (Raizer, Gas Discharge Physics, Springer 1991):
#: air A = 15, B = 365, E/p = 100-800 (also Bazelyan and Raizer, Spark Discharge, p. 32; as
#: 112.5 /(kPa cm) and 2737.5 V/(kPa cm), Husain and Nema, IEEE Trans. Electr. Insul. 1982);
#: N2 12, 342, 100-600; H2 5, 130, 150-600; He 3, 34, 20-150; Ne 4, 100, 100-400; Ar 12, 180,
#: 100-600. Verification: air, Ar and the E/p ranges of air, N2 and H2 were checked against
#: secondary web tables, Ne and the He, Ne and Ar ranges were not found in a source; He B = 34
#: (not 25) follows the tabulations (arXiv:2206.13440, B = 26 /mbar = 34.7; arXiv:1612.07170).
PASCHEN_GASES = {
    name: (a / (1e-2 * 133.322368), b / (1e-2 * 133.322368))
    for name, (a, b) in {
        "air": (15.0, 365.0),
        "N2": (12.0, 342.0),
        "H2": (5.0, 130.0),
        "He": (3.0, 34.0),
        "Ne": (4.0, 100.0),
        "Ar": (12.0, 180.0),
    }.items()
}


def _out(value):
    return np.asarray(value, dtype=float)[()]


def _array(value):
    return np.asarray(value, dtype=float)


# ---------------------------------------------------------------------------------------------
# Sheaths
# ---------------------------------------------------------------------------------------------
def bohm_speed(electron_temperature, ion_temperature=0.0, mass_number=1, charge_number=1, ion_gamma=1.0):
    """Compute the Bohm speed u_B = √((Z T_e + γ_i T_i) / m_i) at the sheath edge.

    Parameters
    ----------
    electron_temperature : float or array_like
        Electron temperature T_e, in eV.
    ion_temperature : float or array_like, optional
        Ion temperature T_i, in eV. Default: ``0`` (cold ions).
    mass_number : float or array_like, optional
        Ion mass in proton masses. Default: ``1``.
    charge_number : float or array_like, optional
        Ion charge number Z. Default: ``1``.
    ion_gamma : float, optional
        Adiabatic index γ_i of the ions. Default: ``1`` (isothermal ions; the fluid Bohm
        criterion with 3 gives the 1-D adiabatic ions of :func:`parameters.sound_speed`).

    Returns
    -------
    float or numpy.ndarray
        u_B, in m/s: the speed the ions have to reach at the sheath edge.

    References
    ----------
    Lieberman and Lichtenberg, section 6.2: u_B = √(e T_e / M) for cold ions.

    Chen, Introduction to Plasma Physics, section 8.2.

    Examples
    --------
    >>> print(f"{bohm_speed(5.0):.4g} m/s")  # hydrogen
    2.188e+04 m/s
    >>> print(f"{bohm_speed(5.0, mass_number=40):.4g} m/s")  # argon
    3460 m/s
    """
    return sound_speed(
        electron_temperature,
        ion_temperature,
        mass_number=mass_number,
        charge_number=charge_number,
        electron_gamma=1.0,
        ion_gamma=ion_gamma,
    )


def bohm_current_density(density, electron_temperature, mass_number=1, charge_number=1, ion_temperature=0.0):
    """Compute the ion current density J = Z e n_s u_B to a wall, with n_s = n exp(−1/2).

    Parameters
    ----------
    density : float or array_like
        Plasma density n far from the wall (the bulk density), in m⁻³.
    electron_temperature : float or array_like
        Electron temperature T_e, in eV.
    mass_number : float or array_like, optional
        Ion mass in proton masses. Default: ``1``.
    charge_number : float or array_like, optional
        Ion charge number Z. Default: ``1``.
    ion_temperature : float or array_like, optional
        Ion temperature T_i, in eV, entering the Bohm speed. Default: ``0``.

    Returns
    -------
    float or numpy.ndarray
        J, in A/m².

    References
    ----------
    Lieberman and Lichtenberg, section 6.2: n_s = n₀ exp(−1/2) ≈ 0.61 n₀ and J = e n_s u_B.

    Examples
    --------
    >>> print(f"{bohm_current_density(1e18, 5.0):.4g} A/m²")
    2127 A/m²
    """
    n = _array(density)
    u_b = bohm_speed(electron_temperature, ion_temperature, mass_number, charge_number)
    return _out(_SHEATH_EDGE * charge_number * _e * n * u_b)


def ion_saturation_current(density, electron_temperature, area, mass_number=1, charge_number=1, ion_temperature=0.0):
    """Compute the ion saturation current I = Z e n_s u_B A of a strongly negative planar probe.

    Valid when the sheath is thin compared with the probe (otherwise see :func:`oml_current`).

    Parameters
    ----------
    density : float or array_like
        Plasma density n, in m⁻³.
    electron_temperature : float or array_like
        Electron temperature T_e, in eV.
    area : float or array_like
        Collecting area A, in m².
    mass_number, charge_number, ion_temperature
        As in :func:`bohm_current_density`.

    Returns
    -------
    float or numpy.ndarray
        I, in A (positive: the magnitude of the ion current).

    References
    ----------
    Chen, Introduction to Plasma Physics, section 8.2.5 (electrostatic probes):
    I_is = 0.61 e n A √(T_e/M).

    Examples
    --------
    >>> # 1 mm² hydrogen probe
    >>> print(f"{ion_saturation_current(1e18, 5.0, 1e-6):.4g} A")
    0.002127 A
    """
    return _out(bohm_current_density(density, electron_temperature, mass_number, charge_number, ion_temperature) * area)


def floating_potential(
    electron_temperature,
    mass_number=1,
    charge_number=1,
    secondary_emission=0.0,
    ion_temperature=0.0,
    electron_mass=electron_mass,
):
    """Compute the floating potential V_f of a wall relative to the plasma.

    The wall charges until the ion current Z Γ_i (at the Bohm flux) equals the net electron
    current (1 − δ) Γ_e of a Maxwellian, Γ_e = n_s √(T_e/(2π m_e)) exp(e V_f / T_e), giving

    V_f = (T_e / 2) ln( Z² (Z + γ_i T_i/T_e) 2π m_e / (m_i (1 − δ)²) ).

    Parameters
    ----------
    electron_temperature : float or array_like
        Electron temperature T_e, in eV.
    mass_number : float or array_like, optional
        Ion mass in proton masses. Default: ``1``.
    charge_number : float or array_like, optional
        Ion charge number Z. Default: ``1``.
    secondary_emission : float or array_like, optional
        Secondary-electron emission coefficient δ, the electrons emitted per incident electron
        (``0 <= δ < 1``). Default: ``0``.
    ion_temperature : float or array_like, optional
        Ion temperature T_i, in eV, entering the Bohm speed (isothermal ions). Default: ``0``.
    electron_mass : float, optional
        Electron mass, in kg. Default: the electron mass.

    Returns
    -------
    float or numpy.ndarray
        V_f, in V, negative: the wall potential minus the plasma potential.

    Raises
    ------
    ValueError
        If ``secondary_emission`` is not below 1 (the wall then emits more than it collects and
        a space-charge-limited sheath forms instead).

    References
    ----------
    Lieberman and Lichtenberg, section 6.2: V_f = −(T_e/2) ln(M/(2π m)) for hydrogen-like
    ions; Hobbs and Wesson, Plasma Phys. Control. Fusion 9, 85 (1967), for the emission factor
    1/(1 − δ).

    Examples
    --------
    >>> print(f"{floating_potential(1.0):.4f} V")  # hydrogen: about −2.84 T_e
    -2.8388 V
    >>> print(f"{floating_potential(1.0, mass_number=40):.4f} V")  # argon
    -4.6832 V
    """
    delta = _array(secondary_emission)
    if np.any(delta >= 1) or np.any(delta < 0):
        raise ValueError("secondary_emission must be in [0, 1)")
    te = _array(electron_temperature)
    u_b = bohm_speed(te, ion_temperature, mass_number, charge_number)
    # ion flux Z u_B against the electron flux (1 − δ) √(e T_e / (2π m_e)) exp(V/T_e)
    electron_thermal_flux = np.sqrt(te * _e / (2 * np.pi * electron_mass))
    return _out(te * np.log(charge_number * u_b / ((1 - delta) * electron_thermal_flux)))


def sheath_width(density, electron_temperature, voltage, model="child-langmuir"):
    """Compute the width of a sheath with the potential drop ``voltage`` to a wall.

    Parameters
    ----------
    density : float or array_like
        Electron (= ion) density n at the sheath edge, in m⁻³.
    electron_temperature : float or array_like
        Electron temperature T_e, in eV, for the Debye length.
    voltage : float or array_like
        Potential drop V₀ across the sheath, in V (its sign doesn't matter).
    model : {"child-langmuir", "matrix"}, optional
        ``"child-langmuir"``: a collisionless, space-charge-limited ion sheath carrying the Bohm
        current, s = (√2/3) λ_D (2 V₀/T_e)^{3/4}; ``"matrix"``: the electrons are gone and the
        ions have the uniform density n, s = √(2 ε₀ V₀/(e n)) = λ_D √(2 V₀/T_e).
        Default: ``"child-langmuir"``.

    Returns
    -------
    float or numpy.ndarray
        The sheath width s, in m.

    Raises
    ------
    ValueError
        If ``model`` is unknown.

    References
    ----------
    Lieberman and Lichtenberg, section 6.3 (the matrix sheath and the Child–Langmuir
    sheath).

    Examples
    --------
    >>> print(f"{sheath_width(1e16, 3.0, 100.0):.4g} m")
    0.001416 m
    >>> print(f"{sheath_width(1e16, 3.0, 100.0, model='matrix'):.4g} m")
    0.001051 m
    """
    ratio = 2 * np.abs(_array(voltage)) / _array(electron_temperature)
    lam = debye_length(density, electron_temperature)
    if model == "child-langmuir":
        return _out(math.sqrt(2) / 3 * lam * ratio**0.75)
    if model == "matrix":
        return _out(lam * np.sqrt(ratio))
    raise ValueError(f"model must be 'child-langmuir' or 'matrix'; got {model!r}")


# ---------------------------------------------------------------------------------------------
# Space-charge-limited flow
# ---------------------------------------------------------------------------------------------
def child_langmuir(voltage, gap, mass=electron_mass, charge=elementary_charge, initial_velocity=0.0):
    """Compute the space-charge-limited current density J of a planar diode.

    For particles starting at rest, J = (4/9) ε₀ √(2|q|/m) V^{3/2} / d². Particles injected with
    the speed v₀ into a space-charge-limited gap (zero field at the emitter) carry

    J = (4/9) ε₀ √(2|q|/m) S (S + 3 w₀)² / d²,  w₀ = v₀ √(m/(2|q|)),  S = √(w₀² + V) − w₀,

    which is the Child–Langmuir law for v₀ = 0 and the space-charge-free 2 ε₀ v₀ V / d² for large
    v₀.

    Parameters
    ----------
    voltage : float or array_like
        Accelerating voltage V across the gap, in V.
    gap : float or array_like
        Gap width d, in m.
    mass : float or array_like, optional
        Particle mass m, in kg. Default: the electron mass.
    charge : float or array_like, optional
        Particle charge q, in C (its sign doesn't matter). Default: e.
    initial_velocity : float or array_like, optional
        Injection speed v₀ along the gap, in m/s. Default: ``0``.

    Returns
    -------
    float or numpy.ndarray
        J, in A/m².

    References
    ----------
    C. D. Child, Phys. Rev. 32, 492 (1911); I. Langmuir, Phys. Rev. 21, 419 (1923).

    Lieberman and Lichtenberg, section 6.3. The injected case follows from integrating
    Poisson's equation once with v² = v₀² + 2|q|φ/m and j = ρ v.

    Examples
    --------
    >>> # 1 keV electrons across 1 cm
    >>> print(f"{child_langmuir(1e3, 1e-2):.4g} A/m²")
    738.1 A/m²
    >>> # H⁻ ions, 10 kV over 1 cm
    >>> print(f"{child_langmuir(1e4, 1e-2, mass=1.00794 * proton_mass):.4g} A/m²")
    542.5 A/m²
    """
    v = np.abs(_array(voltage))
    q = np.abs(_array(charge))
    m = _array(mass)
    speed_scale = np.sqrt(2 * q / m)
    w0 = _array(initial_velocity) / speed_scale
    s = np.sqrt(w0**2 + v) - w0
    return _out(4 / 9 * vacuum_permittivity * speed_scale * s * (s + 3 * w0) ** 2 / np.square(gap))


def _lb_series(geometry, terms):
    """Coefficients a_0, a_1, ... of the Langmuir–Blodgett function f(γ) = Σ a_n γⁿ (β or α).

    Writing the potential as φ = e^{2γ/3} f^{4/3} (cylinders) or φ = f^{4/3} (spheres), Poisson's
    equation with the space-charge-limited current becomes the polynomial equation
    f² + 4 f f' + f'² + 3 f f'' = 1 (cylinders), f'² + 3 f f' + 3 f f'' = 1 (spheres), with
    f(0) = 0, f'(0) = 1. The coefficient of γ^m in it is linear in a_{m+1}, which is solved for
    order by order in exact rational arithmetic.
    """

    def product(p, q, m):
        return sum(p[i] * q[m - i] for i in range(len(p)) if 0 <= m - i < len(q))

    def residual(a, m):
        d = [k * a[k] for k in range(1, len(a))]
        dd = [k * (k - 1) * a[k] for k in range(2, len(a))]
        if geometry == "spherical":
            return product(d, d, m) + 3 * product(a, d, m) + 3 * product(a, dd, m)
        return product(a, a, m) + 4 * product(a, d, m) + product(d, d, m) + 3 * product(a, dd, m)

    a = [Fraction(0), Fraction(1)]
    for m in range(1, terms - 1):
        r0 = residual(a + [Fraction(0)], m)
        r1 = residual(a + [Fraction(1)], m)
        a.append(-r0 / (r1 - r0))
    return [float(c) for c in a[1:]]


# β = γ − 2/5 γ² + 11/120 γ³ − 47/3300 γ⁴ + ... (cylinders), α = γ − 3/10 γ² + 3/40 γ³ − ... (spheres)
_LB_TERMS = 40
_LB_CYLINDRICAL = _lb_series("cylindrical", _LB_TERMS)
_LB_SPHERICAL = _lb_series("spherical", _LB_TERMS)
_LB_MAX_GAMMA = 4.0


def _lb_function(gamma, coefficients):
    result = np.zeros_like(gamma)
    for c in reversed(coefficients):
        result = result * gamma + c
    return result * gamma


def langmuir_blodgett(
    voltage,
    anode_radius,
    cathode_radius,
    geometry="cylindrical",
    mass=electron_mass,
    charge=elementary_charge,
    total=False,
):
    """Compute the space-charge-limited current of coaxial cylinders or concentric spheres.

    With γ = ln(r_a/r_c) the current is, per unit length for cylinders and in total for spheres,

    I = (8π ε₀ / 9) √(2|q|/m) V^{3/2} / (r_a β²)  (cylinders),
    I = (16π ε₀ / 9) √(2|q|/m) V^{3/2} / α²  (spheres),

    with the Langmuir–Blodgett functions β(γ) = γ − 2/5 γ² + 11/120 γ³ − ... and
    α(γ) = γ − 3/10 γ² + 3/40 γ³ − ..., summed to γ¹⁰. For r_a/r_c → 1 both reduce to the planar
    :func:`child_langmuir` law with d = r_a − r_c. The emitter may be inside (r_a > r_c) or
    outside (r_a < r_c, γ < 0).

    Parameters
    ----------
    voltage : float or array_like
        Voltage V between emitter and collector, in V.
    anode_radius : float or array_like
        Collector radius r_a, in m.
    cathode_radius : float or array_like
        Emitter radius r_c, in m.
    geometry : {"cylindrical", "spherical"}, optional
        Default: ``"cylindrical"``.
    mass : float or array_like, optional
        Particle mass m, in kg. Default: the electron mass.
    charge : float or array_like, optional
        Particle charge q, in C (its sign doesn't matter). Default: e.
    total : bool, optional
        ``False``: the current density at the collector in A/m² (comparable to
        :func:`child_langmuir`); ``True``: the current per unit length in A/m (cylinders) or
        the total current in A (spheres). Default: ``False``.

    Returns
    -------
    float or numpy.ndarray
        The current density in A/m², or the current as selected by ``total``.

    Raises
    ------
    ValueError
        If ``geometry`` is unknown or |γ| exceeds 4 (r_a/r_c outside about 1/55 to 55), where
        the series isn't converged.

    References
    ----------
    I. Langmuir and K. B. Blodgett, Phys. Rev. 22, 347 (1923) (cylinders) and Phys. Rev. 24, 49
    (1924) (spheres). The series coefficients are re-derived from their equation
    φ'' = k r φ^{-1/2} (cylinders), φ'' + φ' = k φ^{-1/2} (spheres), in γ.

    Examples
    --------
    >>> # electrons, 1 kV, anode radius 5 mm, cathode radius 1 mm
    >>> print(f"{langmuir_blodgett(1e3, 5e-3, 1e-3, total=True):.4g} A/m")
    121 A/m
    >>> print(
    ...     f"{langmuir_blodgett(1e3, 5e-3, 1e-3, geometry='spherical', total=True):.4g} A"
    ... )
    0.8126 A
    """
    if geometry not in ("cylindrical", "spherical"):
        raise ValueError(f"geometry must be 'cylindrical' or 'spherical'; got {geometry!r}")
    r_a = _array(anode_radius)
    gamma = np.log(r_a / _array(cathode_radius))
    if np.any(np.abs(gamma) > _LB_MAX_GAMMA):
        raise ValueError("the Langmuir–Blodgett series needs |ln(r_a / r_c)| <= 4")
    prefactor = (
        vacuum_permittivity * np.sqrt(2 * np.abs(_array(charge)) / _array(mass)) * np.abs(_array(voltage)) ** 1.5
    )
    if geometry == "cylindrical":
        beta = _lb_function(gamma, _LB_CYLINDRICAL)
        current = 8 * np.pi / 9 * prefactor / (r_a * beta**2)
        return _out(current if total else current / (2 * np.pi * r_a))
    alpha = _lb_function(gamma, _LB_SPHERICAL)
    current = 16 * np.pi / 9 * prefactor / alpha**2
    return _out(current if total else current / (4 * np.pi * r_a**2))


# ---------------------------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------------------------
def _erfcx(x):
    """exp(x²) erfc(x) for x >= 0, without overflow."""
    x = np.asarray(x, dtype=float)
    small = np.where(x < 25, x, 0.0)
    exact = np.exp(small**2) * np.vectorize(math.erfc, otypes=[float])(small)
    big = np.where(x < 25, 25.0, x)
    asymptotic = (1 - 1 / (2 * big**2) + 3 / (4 * big**4) - 15 / (8 * big**6)) / (big * math.sqrt(math.pi))
    return np.where(x < 25, exact, asymptotic)


def oml_current(
    density,
    temperature,
    potential,
    radius,
    length=None,
    geometry="cylindrical",
    mass=electron_mass,
    charge=-elementary_charge,
):
    """Compute the orbital-motion-limited (OML) current collected by a probe.

    For a Maxwellian species with the random current I₀ = A n |q| √(T/(2π m)) and
    χ = −q V/T (positive when the probe attracts the species):

    * repelled (χ ≤ 0): I = I₀ exp(χ);
    * attracted sphere: I = I₀ (1 + χ);
    * attracted cylinder: I = I₀ [exp(χ) erfc(√χ) + (2/√π) √χ].

    The OML limit holds when the sheath is thick enough for no orbit to be blocked by the
    angular momentum barrier, for probe radii of a few Debye lengths at most.

    Parameters
    ----------
    density : float or array_like
        Density n of the collected species, in m⁻³.
    temperature : float or array_like
        Temperature T of the collected species, in eV.
    potential : float or array_like
        Probe potential V relative to the plasma, in V.
    radius : float or array_like
        Probe radius, in m.
    length : float or array_like, optional
        Probe length (cylinders only, end effects ignored), in m.
    geometry : {"cylindrical", "spherical"}, optional
        Default: ``"cylindrical"``.
    mass : float or array_like, optional
        Mass of the collected species, in kg. Default: the electron mass.
    charge : float or array_like, optional
        Charge q of the collected species, in C. Default: −e (electrons; for ions pass the ion
        mass and +e).

    Returns
    -------
    float or numpy.ndarray
        The magnitude of the collected current, in A.

    Raises
    ------
    ValueError
        If ``geometry`` is unknown, or a cylinder is given without ``length``.

    References
    ----------
    H. M. Mott-Smith and I. Langmuir, Phys. Rev. 28, 727 (1926); Chen, Introduction to Plasma
    Physics, section 8.2.5; the exact cylindrical Maxwellian result is in J. G. Laframboise,
    UTIAS Report 100 (1966).

    Examples
    --------
    >>> # a 1 mm thick, 1 cm long wire at +50 V in a 1e14 m⁻³, 2 eV plasma: electron current
    >>> print(f"{oml_current(1e14, 2.0, 50.0, 0.5e-3, length=1e-2):.4g} A")
    0.0006851 A
    >>> # the same wire at −10 V collects ions (hydrogen) in the OML limit
    >>> print(
    ...     f"{oml_current(1e14, 2.0, -10.0, 0.5e-3, 1e-2, mass=proton_mass, charge=elementary_charge):.4g} A"
    ... )
    7.658e-06 A
    """
    if geometry == "cylindrical":
        if length is None:
            raise ValueError("a cylindrical probe needs its length")
        area = 2 * np.pi * _array(radius) * _array(length)
    elif geometry == "spherical":
        area = 4 * np.pi * _array(radius) ** 2
    else:
        raise ValueError(f"geometry must be 'cylindrical' or 'spherical'; got {geometry!r}")
    q = _array(charge)
    t = _array(temperature)
    random_current = area * _array(density) * np.abs(q) * np.sqrt(t * ev_to_joule / (2 * np.pi * _array(mass)))
    chi = -q * _array(potential) / (t * ev_to_joule)
    attracted = np.maximum(chi, 0.0)
    if geometry == "cylindrical":
        # exp(χ) erfc(√χ) = erfcx(√χ)
        gain = _erfcx(np.sqrt(attracted)) + 2 / math.sqrt(math.pi) * np.sqrt(attracted)
    else:
        gain = np.where(chi > 0, 1 + attracted, 1.0)
    return _out(random_current * np.where(chi > 0, gain, np.exp(np.minimum(chi, 0.0))))


# ---------------------------------------------------------------------------------------------
# Transport and breakdown
# ---------------------------------------------------------------------------------------------
def ambipolar_diffusion(ion_mobility, electron_temperature, ion_temperature=0.0, electron_mobility=None):
    """Compute the ambipolar diffusion coefficient D_a = (μ_i D_e + μ_e D_i)/(μ_i + μ_e).

    With the Einstein relations D = μ T/e this is D_a = μ_i μ_e (T_e + T_i)/(μ_i + μ_e), which
    for μ_e ≫ μ_i is D_a = μ_i (T_e + T_i) = D_i (1 + T_e/T_i).

    Parameters
    ----------
    ion_mobility : float or array_like
        Ion mobility μ_i, in m²/(V s).
    electron_temperature : float or array_like
        Electron temperature T_e, in eV.
    ion_temperature : float or array_like, optional
        Ion temperature T_i, in eV. Default: ``0``.
    electron_mobility : float or array_like, optional
        Electron mobility μ_e, in m²/(V s). Default: ``None``, the limit μ_e ≫ μ_i.

    Returns
    -------
    float or numpy.ndarray
        D_a, in m²/s.

    References
    ----------
    Lieberman and Lichtenberg, section 5.1; Chen, chapter 5.

    Examples
    --------
    >>> # N₂⁺ ions in N₂ (μ_i ≈ 2e-4 m²/Vs at 1 Torr), T_e = 3 eV, T_i = 0.03 eV
    >>> print(f"{ambipolar_diffusion(2e-4, 3.0, 0.03):.4g} m²/s")
    0.000606 m²/s
    """
    mu_i = _array(ion_mobility)
    te = _array(electron_temperature)
    ti = _array(ion_temperature)
    if electron_mobility is None:
        return _out(mu_i * (te + ti))
    mu_e = _array(electron_mobility)
    return _out(mu_i * mu_e * (te + ti) / (mu_i + mu_e))


def bohm_diffusion(electron_temperature, field):
    """Compute the Bohm diffusion coefficient D_B = T_e / (16 e B).

    Parameters
    ----------
    electron_temperature : float or array_like
        Electron temperature T_e, in eV.
    field : float or array_like
        Magnetic field B, in T.

    Returns
    -------
    float or numpy.ndarray
        D_B, in m²/s.

    References
    ----------
    D. Bohm, in The Characteristics of Electrical Discharges in Magnetic Fields (McGraw-Hill,
    1949); Chen, Introduction to Plasma Physics, chapter 5 (Bohm diffusion): D_B = k T_e / (16 e B).

    Examples
    --------
    >>> print(f"{bohm_diffusion(10.0, 0.01):.4g} m²/s")
    62.5 m²/s
    """
    return _out(_array(electron_temperature) / (16 * np.abs(_array(field))))


def paschen_breakdown(pressure, gap, gas="air", secondary_emission=0.01, constants=None):
    """Compute the breakdown voltage V_b = B p d / (ln(A p d) − ln ln(1 + 1/γ)).

    Parameters
    ----------
    pressure : float or array_like
        Gas pressure p, in Pa.
    gap : float or array_like
        Electrode gap d, in m.
    gas : str, optional
        A key of :data:`PASCHEN_GASES` (``"air"``, ``"N2"``, ``"H2"``, ``"He"``, ``"Ne"``,
        ``"Ar"``). Default: ``"air"``.
    secondary_emission : float or array_like, optional
        Secondary-emission coefficient γ of the cathode. Default: ``0.01``.
    constants : tuple of float, optional
        ``(A, B)`` in 1/(Pa m) and V/(Pa m), overriding ``gas``.

    Returns
    -------
    float or numpy.ndarray
        V_b, in V. ``nan`` where p d is too small for any voltage to break down (the
        denominator is not positive: left of the Paschen minimum's asymptote).

    Raises
    ------
    ValueError
        If ``gas`` is unknown.

    Notes
    -----
    A and B are fits of the Townsend coefficient, so the formula is only accurate when the
    breakdown field E/p = V_b/(p d) is inside the range of the gas. The minimum is
    V_min = e (B/A) ln(1 + 1/γ) at p d = e ln(1 + 1/γ)/A, which for γ = 0.01 is within about
    10 % of the measured minimum sparking potential for air and He but 20-40 % too high for
    H2, N2 and Ar, and its p d is off by 5-120 % (measured air: 327 V at 0.567 Torr cm; N2:
    251 V at 0.67; H2: 273 V at 1.15; He: 156 V at 4.0; Ar: 137 V at 0.9; Naidu, High Voltage
    Engineering, p. 27). Use ``secondary_emission`` to fit a measured minimum.

    References
    ----------
    F. Paschen, Ann. Phys. 273, 69 (1889); Raizer, Gas Discharge Physics (Springer, 1991). The constants and their
    E/p validity ranges are documented at :data:`PASCHEN_GASES`.

    Examples
    --------
    >>> # air, 1 atm, 1 mm gap
    >>> print(f"{paschen_breakdown(101325.0, 1e-3):.4g} V")
    5035 V
    >>> # too small a p d: no breakdown
    >>> print(paschen_breakdown(1.0, 1e-3))
    nan
    """
    if constants is None:
        if gas not in PASCHEN_GASES:
            raise ValueError(f"gas must be one of {list(PASCHEN_GASES)}; got {gas!r}")
        constants = PASCHEN_GASES[gas]
    a, b = constants
    pd = _array(pressure) * _array(gap)
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = np.log(a * pd) - np.log(np.log1p(1 / _array(secondary_emission)))
        return _out(np.where(denominator > 0, b * pd / denominator, np.nan))


def mean_free_path(cross_section, density=None, pressure=None, temperature=300.0):
    """Compute the mean free path λ = 1/(n σ) of a particle in a gas.

    Give either the gas density or its pressure.

    Parameters
    ----------
    cross_section : float or array_like
        Collision cross section σ, in m².
    density : float or array_like, optional
        Gas number density n, in m⁻³.
    pressure : float or array_like, optional
        Gas pressure p, in Pa; then n = p/(k_B T).
    temperature : float or array_like, optional
        Gas temperature T, in **K** (not eV; the gas is cold), used with ``pressure``.
        Default: ``300``.

    Returns
    -------
    float or numpy.ndarray
        λ, in m.

    Raises
    ------
    ValueError
        If not exactly one of ``density`` and ``pressure`` is given.

    References
    ----------
    Lieberman and Lichtenberg, section 3.1: λ = 1/(n σ).

    Examples
    --------
    >>> # an electron in argon at 1 Pa, 300 K, with σ = 1e-19 m² (about 1 eV)
    >>> print(f"{mean_free_path(1e-19, pressure=1.0):.4g} m")
    0.04142 m
    """
    if (density is None) == (pressure is None):
        raise ValueError("give exactly one of density and pressure")
    if density is None:
        density = _array(pressure) / (boltzmann_constant * _array(temperature))
    return _out(1 / (_array(density) * _array(cross_section)))
