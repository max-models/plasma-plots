"""Fluid, MHD and cold-plasma waves: light waves, MHD waves at any angle, dissipative and Hall-MHD
Alfvén waves, cold-plasma (Stix) waves with their cutoffs and resonances, Faraday rotation, cavity
modes, drift waves and Hasegawa–Wakatani, Alfvén and slow continua and the TAE frequency.

All frequencies are complex for ``exp(i(k·x − ωt))`` (a positive imaginary part is growth) and are
the positive-frequency branches (−ω* is a solution as well). Characteristic frequencies that are
not functions of k (cutoffs, resonances, cavity modes, the TAE frequency) are real. Several branches
come as dicts of branch names to frequencies, which ``array.struphy.plot.dispersion(branches=...)``
takes as they are. Units are whatever the speeds, lengths and frequencies are given in.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.polynomial import polynomial as _poly

__all__ = [
    "Species",
    "alfven_continuum",
    "appleton_hartree",
    "cavity_modes",
    "cold_plasma_waves",
    "cutoffs",
    "dissipative_alfven",
    "drift_wave",
    "electron_ion",
    "faraday_rotation",
    "group_velocity",
    "hall_mhd_parallel",
    "hasegawa_wakatani",
    "light_wave",
    "magnetosonic_speeds",
    "mhd_waves",
    "parallel_wavenumber",
    "plasma_light_wave",
    "refractive_index",
    "resonances",
    "slow_continuum",
    "stix",
    "tae_frequency",
]


def _complex(x):
    return np.asarray(x, dtype=complex)[()]


def _of_r(value, r):
    """A profile given as a number, an array or a callable of r, evaluated at r."""
    return np.asarray(value(r) if callable(value) else value, dtype=float)


def _companion_roots(coefficients):
    """The (real parts of the) roots of polynomials with ascending coefficients along the last axis."""
    d = coefficients.shape[-1] - 1
    companion = np.zeros(coefficients.shape[:-1] + (d, d))
    companion[..., np.arange(1, d), np.arange(d - 1)] = 1.0
    companion[..., :, -1] = -coefficients[..., :-1] / coefficients[..., -1:]
    return np.linalg.eigvals(companion).real


def _sorted_real_roots(coefficients):
    """The roots of polynomials with only real roots, sorted ascending.

    ``coefficients`` has shape (..., d + 1), ascending powers, the leading one nonzero. Roots
    smaller than the geometric mean of all come from the reversed polynomial (as reciprocals of its
    roots), which keeps their relative accuracy when the roots span many orders of magnitude.
    """
    coefficients = np.asarray(coefficients, dtype=float)
    d = coefficients.shape[-1] - 1
    roots = np.sort(_companion_roots(coefficients), axis=-1)
    constant = coefficients[..., 0]
    usable = constant != 0
    reversed_ = np.where(usable[..., None], coefficients[..., ::-1], coefficients)
    with np.errstate(divide="ignore", invalid="ignore"):
        small = np.sort(1 / _companion_roots(reversed_), axis=-1)
        scale = np.abs(constant / coefficients[..., -1]) ** (1 / d)
    use = usable[..., None] & (np.abs(small) < scale[..., None])
    return np.sort(np.where(use, small, roots), axis=-1)


# ----------------------------------------------------------------------------------------------
# Unmagnetized plasma
# ----------------------------------------------------------------------------------------------


def light_wave(k, c=1.0):
    """Compute the frequency ω = c|k| of a light wave in vacuum.

    Parameters
    ----------
    k : float or array_like
        The wavenumber.
    c : float or array_like, optional
        The speed of light. Default: ``1.0``.

    Returns
    -------
    complex or numpy.ndarray
        ω = c|k|, complex.

    References
    ----------
    J. D. Jackson, Classical Electrodynamics, 3rd ed. (Wiley, 1999).

    Examples
    --------
    >>> print(np.round(light_wave(np.array([-1.0, 0.5, 2.0])).real, 4))
    [1.  0.5 2. ]
    """
    return _complex(np.asarray(c) * np.abs(k))


def plasma_light_wave(k, plasma_frequency=1.0, c=1.0):
    """Compute the frequency of a light wave in an unmagnetized cold plasma, ω² = ω_p² + c²k².

    This is also the ordinary (O) mode of a magnetized plasma, propagating perpendicular to B.
    Waves below the cutoff ω = ω_p do not propagate.

    Parameters
    ----------
    k : float or array_like
        The wavenumber.
    plasma_frequency : float or array_like, optional
        The (total) plasma frequency ω_p. Default: ``1.0``.
    c : float or array_like, optional
        The speed of light. Default: ``1.0``.

    Returns
    -------
    complex or numpy.ndarray
        ω = √(ω_p² + c²k²), complex.

    References
    ----------
    F. F. Chen, Introduction to Plasma Physics and Controlled Fusion, 3rd ed. (Springer, 2016),
    ch. 4.

    Examples
    --------
    >>> print(round(float(plasma_light_wave(1.0, plasma_frequency=1.0).real), 6))   # √2
    1.414214
    """
    k, wp, c = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (k, plasma_frequency, c)))
    return _complex(np.sqrt(wp**2 + c**2 * k**2))


# ----------------------------------------------------------------------------------------------
# Ideal and extended MHD
# ----------------------------------------------------------------------------------------------


def magnetosonic_speeds(theta=0.0, alfven_speed=1.0, sound_speed=0.5):
    """Compute the phase speeds of the three ideal-MHD waves (the Friedrichs diagram).

    With the angle θ between k and B₀:

    * shear Alfvén: v = v_A |cos θ|,
    * fast and slow magnetosonic: v² = ½ [c_s² + v_A² ± √((c_s² + v_A²)² − 4 c_s² v_A² cos²θ)].

    So v_f² + v_s² = c_s² + v_A², v_f² v_s² = c_s² v_A² cos²θ and v_s ≤ v_A|cos θ| ≤ v_f.

    Parameters
    ----------
    theta : float or array_like, optional
        The angle between k and B₀, in radians. Default: ``0.0``.
    alfven_speed : float or array_like, optional
        The Alfvén speed v_A = B₀/√(μ₀ρ₀). Default: ``1.0``.
    sound_speed : float or array_like, optional
        The sound speed c_s = √(γp₀/ρ₀). Default: ``0.5``.

    Returns
    -------
    dict of str to float or numpy.ndarray
        ``{"shear Alfvén": ..., "slow": ..., "fast": ...}``, the phase speeds ω/k (real, ≥ 0).

    References
    ----------
    J. P. Goedbloed and S. Poedts, Principles of Magnetohydrodynamics (Cambridge University Press,
    2004), ch. 5.
    J. P. Freidberg, Ideal MHD (Cambridge University Press, 2014).

    Examples
    --------
    >>> v = magnetosonic_speeds(np.pi / 2, alfven_speed=1.0, sound_speed=0.5)
    >>> print({name: round(float(s), 4) for name, s in v.items()})
    {'shear Alfvén': 0.0, 'slow': 0.0, 'fast': 1.118}
    """
    theta, va, cs = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (theta, alfven_speed, sound_speed)))
    cos2 = np.cos(theta) ** 2
    total = va**2 + cs**2
    fast2 = (total + np.sqrt(np.maximum(total**2 - 4 * va**2 * cs**2 * cos2, 0.0))) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        slow2 = np.where(fast2 > 0, va**2 * cs**2 * cos2 / fast2, 0.0)
    return {
        "shear Alfvén": (va * np.abs(np.cos(theta)))[()],
        "slow": np.sqrt(slow2)[()],
        "fast": np.sqrt(fast2)[()],
    }


def mhd_waves(k, theta=0.0, alfven_speed=1.0, sound_speed=0.5):
    """Compute the frequencies of the shear Alfvén, slow and fast waves of ideal MHD.

    ω = |k| v with the phase speeds v of :func:`magnetosonic_speeds`: a homogeneous plasma with a
    straight field B₀, the wavevector at the angle θ to B₀. This is Struphy's ``LinearMHD`` (and
    ``MHDhomogenSlab`` in ``struphy.dispersion_relations``, where cos θ = B₀z/|B₀|).

    Parameters
    ----------
    k : float or array_like
        The wavenumber |k|.
    theta : float or array_like, optional
        The angle between k and B₀, in radians. Default: ``0.0``.
    alfven_speed : float or array_like, optional
        The Alfvén speed v_A. Default: ``1.0``.
    sound_speed : float or array_like, optional
        The sound speed c_s = √(γp₀/ρ₀). Default: ``0.5``.

    Returns
    -------
    dict of str to complex or numpy.ndarray
        ``{"shear Alfvén": ..., "slow": ..., "fast": ...}``, complex frequencies.

    See Also
    --------
    magnetosonic_speeds : The phase speeds.

    References
    ----------
    J. P. Goedbloed and S. Poedts, Principles of Magnetohydrodynamics (Cambridge University Press,
    2004), ch. 5.

    Examples
    --------
    >>> w = mhd_waves(2.0, theta=np.pi / 3, alfven_speed=1.0, sound_speed=0.5)
    >>> print({name: round(float(o.real), 4) for name, o in w.items()})
    {'shear Alfvén': 1.0, 'slow': 0.4569, 'fast': 2.1889}
    """
    speeds = magnetosonic_speeds(theta, alfven_speed, sound_speed)
    return {name: _complex(np.abs(k) * v) for name, v in speeds.items()}


def dissipative_alfven(k, alfven_speed=1.0, resistivity=0.0, viscosity=0.0, theta=0.0):
    """Compute the frequencies of shear Alfvén waves with resistivity and viscosity.

    The incompressible shear Alfvén wave of visco-resistive MHD,
    ∂u/∂t = v_A ∂b/∂z + ν∇²u and ∂b/∂t = v_A ∂u/∂z + η∇²b (b in velocity units), gives

    ω = −i (η + ν) k²/2 ± √(k∥² v_A² − (η − ν)² k⁴/4),   k∥ = k cos θ.

    With η = ν the damping rate is exactly (η + ν)k²/2; for (η − ν)²k⁴/4 > k∥²v_A² the wave is
    overdamped (Re ω = 0).

    Parameters
    ----------
    k : float or array_like
        The wavenumber |k|.
    alfven_speed : float or array_like, optional
        The Alfvén speed v_A. Default: ``1.0``.
    resistivity : float or array_like, optional
        The magnetic diffusivity η. Default: ``0.0``.
    viscosity : float or array_like, optional
        The kinematic viscosity ν. Default: ``0.0``.
    theta : float or array_like, optional
        The angle between k and B₀, in radians. Default: ``0.0``.

    Returns
    -------
    dict of str to complex or numpy.ndarray
        ``{"forward": ..., "backward": ...}``, the + and − roots (principal square root, so
        ``"forward"`` has Re ω ≥ 0 and is the less damped root when the wave is overdamped).

    References
    ----------
    J. P. Goedbloed and S. Poedts, Principles of Magnetohydrodynamics (Cambridge University Press,
    2004).

    Examples
    --------
    >>> w = dissipative_alfven(1.0, resistivity=0.1, viscosity=0.1)["forward"]
    >>> print(round(float(w.real), 4), round(float(w.imag), 4))
    1.0 -0.1
    """
    k = np.asarray(k, dtype=float)
    eta, nu = np.asarray(resistivity, dtype=float), np.asarray(viscosity, dtype=float)
    k_par = k * np.cos(theta)
    root = np.sqrt((k_par**2 * np.asarray(alfven_speed) ** 2 - (eta - nu) ** 2 * k**4 / 4).astype(complex))
    damping = -0.5j * (eta + nu) * k**2
    return {"forward": _complex(damping + root), "backward": _complex(damping - root)}


def hall_mhd_parallel(k, alfven_speed=1.0, ion_inertial_length=1.0):
    """Compute the whistler and ion-cyclotron branches of Hall MHD along B₀.

    Parallel to B₀ the Hall term splits the shear Alfvén wave into the right-hand (whistler) and
    left-hand (ion-cyclotron) polarized waves,

    ω = |k| v_A (√(1 + k²d_i²/4) ± |k| d_i/2).

    The whistler goes to ω → k² v_A d_i at large k d_i, the ion-cyclotron wave to the ion cyclotron
    frequency Ω_i = v_A/d_i. (The sound wave ω = |k| c_s decouples.)

    Parameters
    ----------
    k : float or array_like
        The wavenumber along B₀.
    alfven_speed : float or array_like, optional
        The Alfvén speed v_A. Default: ``1.0``.
    ion_inertial_length : float or array_like, optional
        The ion inertial length d_i = c/ω_pi = v_A/Ω_i. Default: ``1.0``.

    Returns
    -------
    dict of str to complex or numpy.ndarray
        ``{"whistler": ..., "ion cyclotron": ...}``, complex frequencies.

    References
    ----------
    E. Hameiri, A. Ishizawa and A. Ishida, "Waves in the Hall-magnetohydrodynamics model",
    Phys. Plasmas 12, 072109 (2005).

    Examples
    --------
    >>> w = hall_mhd_parallel(1.0, alfven_speed=1.0, ion_inertial_length=1.0)
    >>> print({name: round(float(o.real), 4) for name, o in w.items()})
    {'whistler': 1.618, 'ion cyclotron': 0.618}
    """
    k = np.abs(np.asarray(k, dtype=float))
    va, d = np.asarray(alfven_speed, dtype=float), np.asarray(ion_inertial_length, dtype=float)
    root = np.sqrt(1 + (k * d) ** 2 / 4)
    return {"whistler": _complex(k * va * (root + k * d / 2)), "ion cyclotron": _complex(k * va * (root - k * d / 2))}


# ----------------------------------------------------------------------------------------------
# Cold plasma
# ----------------------------------------------------------------------------------------------


class Species(NamedTuple):
    """One species of a cold plasma, by its plasma frequency and its signed cyclotron frequency.

    Any ``(plasma_frequency, cyclotron_frequency)`` pair works in place of a ``Species``. The
    cyclotron frequency carries the sign of the charge, Ω_s = q_s B₀/m_s (negative for electrons);
    the plasma frequency is ω_ps = √(n_s q_s²/(ε₀ m_s)). Both in the same (arbitrary) units as the
    wave frequency, e.g. normalized to |Ω_e| or ω_pe. :func:`electron_ion` builds a quasi-neutral
    electron–ion pair.

    Parameters
    ----------
    plasma_frequency : float
        ω_ps ≥ 0.
    cyclotron_frequency : float
        Ω_s, signed: negative for negative charges.

    Attributes
    ----------
    plasma_frequency : float
        ω_ps.
    cyclotron_frequency : float
        Ω_s, signed.

    Examples
    --------
    >>> Species(plasma_frequency=2.0, cyclotron_frequency=-1.0)
    Species(plasma_frequency=2.0, cyclotron_frequency=-1.0)
    """

    plasma_frequency: float
    cyclotron_frequency: float


def electron_ion(plasma_frequency=1.0, cyclotron_frequency=1.0, mass_ratio=1836.15267343, charge=1):
    """Build the species of a quasi-neutral electron–ion plasma from the electron frequencies.

    With the ion charge number Z and mass ratio μ = m_i/m_e, quasi-neutrality n_i = n_e/Z gives
    ω_pi² = ω_pe² Z/μ and Ω_i = Z|Ω_e|/μ.

    Parameters
    ----------
    plasma_frequency : float, optional
        The electron plasma frequency ω_pe. Default: ``1.0``.
    cyclotron_frequency : float, optional
        The electron cyclotron frequency |Ω_e| (its sign is ignored). Default: ``1.0``.
    mass_ratio : float, optional
        m_i/m_e. Default: the proton, ``1836.15267343``.
    charge : float, optional
        The ion charge number Z. Default: ``1``.

    Returns
    -------
    list of Species
        ``[electrons, ions]``, electrons with a negative cyclotron frequency.

    Examples
    --------
    >>> electrons, ions = electron_ion(2.0, 1.0, mass_ratio=100.0)
    >>> print(electrons.cyclotron_frequency, round(ions.plasma_frequency, 4), ions.cyclotron_frequency)
    -1.0 0.2 0.01
    """
    wce = abs(float(cyclotron_frequency))
    return [
        Species(float(plasma_frequency), -wce),
        Species(float(plasma_frequency) * np.sqrt(charge / mass_ratio), charge * wce / mass_ratio),
    ]


def _species_list(species):
    """A list of (ω_p, Ω) pairs from one species or a sequence of species."""
    if isinstance(species, Species) or (len(species) == 2 and all(np.ndim(s) == 0 for s in species)):
        species = [species]
    return [(np.asarray(s[0], dtype=float), np.asarray(s[1], dtype=float)) for s in species]


def stix(omega, species):
    """Compute Stix's cold-plasma dielectric parameters S, D, P, R and L.

    R = 1 − Σ_s ω_ps²/(ω(ω + Ω_s)), L = 1 − Σ_s ω_ps²/(ω(ω − Ω_s)), P = 1 − Σ_s ω_ps²/ω²,
    S = (R + L)/2 and D = (R − L)/2, with signed Ω_s (so R is resonant at the electron cyclotron
    frequency). The dielectric tensor is ((S, −iD, 0), (iD, S, 0), (0, 0, P)) with B₀ along z.

    Parameters
    ----------
    omega : float, complex or array_like
        The wave frequency.
    species : Species, (float, float) or sequence of these
        The species: plasma frequency and signed cyclotron frequency of each.

    Returns
    -------
    dict of str to float or numpy.ndarray
        ``{"S": ..., "D": ..., "P": ..., "R": ..., "L": ...}``.

    References
    ----------
    T. H. Stix, Waves in Plasmas (American Institute of Physics, 1992), ch. 1.

    Examples
    --------
    >>> s = stix(2.0, Species(1.0, -1.0))     # electrons, ω = 2 ω_pe = 2 |Ω_e|
    >>> print({name: round(float(v), 4) for name, v in s.items()})
    {'S': 0.6667, 'D': -0.1667, 'P': 0.75, 'R': 0.5, 'L': 0.8333}
    """
    omega = np.asarray(omega)
    omega = omega.astype(complex if np.iscomplexobj(omega) else float)
    r = l = p = 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        for wp, wc in _species_list(species):
            r = r - wp**2 / (omega * (omega + wc))
            l = l - wp**2 / (omega * (omega - wc))
            p = p - wp**2 / omega**2
    r, l, p = (np.asarray(v)[()] for v in np.broadcast_arrays(r, l, p))
    return {"S": (r + l) / 2, "D": (r - l) / 2, "P": p, "R": r, "L": l}


def refractive_index(omega, theta, species):
    """Compute the two squared refractive indices n² = c²k²/ω² of cold-plasma waves.

    The roots of Stix's biquadratic A n⁴ − B n² + C = 0 with A = S sin²θ + P cos²θ,
    B = RL sin²θ + PS(1 + cos²θ), C = PRL:

    n² = (B ± F)/(2A),   F² = (RL − PS)² sin⁴θ + 4P²D² cos²θ.

    Along B₀ (θ = 0) the roots are R and L, across it (θ = π/2) RL/S (X mode) and P (O mode), in an
    order set by the signs. n² < 0 is an evanescent wave, n² = 0 a cutoff and n² → ∞ a resonance.

    Parameters
    ----------
    omega : float, complex or array_like
        The wave frequency.
    theta : float or array_like
        The angle between k and B₀, in radians.
    species : Species, (float, float) or sequence of these
        The species: plasma frequency and signed cyclotron frequency of each.

    Returns
    -------
    (float or numpy.ndarray, float or numpy.ndarray)
        n² with the + and with the − sign.

    See Also
    --------
    stix : The parameters S, D, P, R, L.
    appleton_hartree : The same for electrons only, as O and X modes.

    References
    ----------
    T. H. Stix, Waves in Plasmas (American Institute of Physics, 1992), ch. 1.

    Examples
    --------
    >>> plus, minus = refractive_index(2.0, np.pi / 2, Species(1.0, -1.0))
    >>> print(round(float(plus), 4), round(float(minus), 4))     # O mode P, X mode RL/S
    0.75 0.625
    """
    s = stix(omega, species)
    S, D, P, R, L = (s[name] for name in "SDPRL")
    sin2, cos2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    a = S * sin2 + P * cos2
    b = R * L * sin2 + P * S * (1 + cos2)
    f = np.sqrt((R * L - P * S) ** 2 * sin2**2 + 4 * P**2 * D**2 * cos2)
    c = P * R * L
    with np.errstate(divide="ignore", invalid="ignore"):
        # the root without cancellation directly, the other one from the product C/A
        positive = np.real(b) >= 0
        plus = np.where(positive, (b + f) / (2 * a), 2 * c / (b - f))
        minus = np.where(positive, 2 * c / (b + f), (b - f) / (2 * a))
    return plus[()], minus[()]


def appleton_hartree(omega, theta, plasma_frequency=1.0, cyclotron_frequency=1.0):
    """Compute the Appleton–Hartree refractive indices of a cold magnetized electron plasma.

    With X = ω_pe²/ω² and Y = |Ω_e|/ω (ions immobile),

    n² = 1 − X(1 − X) / (1 − X − ½Y² sin²θ ± √(¼Y⁴ sin⁴θ + (1 − X)² Y² cos²θ)),

    the upper sign the ordinary (O) and the lower the extraordinary (X) mode. Across B₀ they are
    n² = 1 − X and n² = 1 − X(1 − X)/(1 − X − Y²); along B₀ (for X < 1) the L and R waves.

    Parameters
    ----------
    omega : float or array_like
        The wave frequency.
    theta : float or array_like
        The angle between k and B₀, in radians.
    plasma_frequency : float or array_like, optional
        The electron plasma frequency ω_pe. Default: ``1.0``.
    cyclotron_frequency : float or array_like, optional
        The electron cyclotron frequency |Ω_e|. Default: ``1.0``.

    Returns
    -------
    dict of str to float or numpy.ndarray
        ``{"O": n²_O, "X": n²_X}``.

    See Also
    --------
    refractive_index : Any number of species.

    References
    ----------
    T. H. Stix, Waves in Plasmas (American Institute of Physics, 1992), ch. 1.
    F. F. Chen, Introduction to Plasma Physics and Controlled Fusion, 3rd ed. (Springer, 2016),
    ch. 4.

    Examples
    --------
    >>> n2 = appleton_hartree(2.0, np.pi / 2, plasma_frequency=1.0, cyclotron_frequency=1.0)
    >>> print({name: round(float(v), 4) for name, v in n2.items()})
    {'O': 0.75, 'X': 0.625}
    """
    omega = np.asarray(omega, dtype=float)
    x = np.asarray(plasma_frequency, dtype=float) ** 2 / omega**2
    y = np.abs(np.asarray(cyclotron_frequency, dtype=float)) / omega
    sin2, cos2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    root = np.sqrt(y**4 * sin2**2 / 4 + (1 - x) ** 2 * y**2 * cos2)
    base = 1 - x - y**2 * sin2 / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "O": np.asarray(1 - x * (1 - x) / (base + root))[()],
            "X": np.asarray(1 - x * (1 - x) / (base - root))[()],
        }


def _cold_polynomials(species):
    """Polynomials in x = ω² (ascending coefficients) of the cold-plasma dispersion relation.

    With Q = Π(x − Ω_s²): Sp = xQS, Pp = x − Σω_p² = xP and G = x²QRL.
    """
    species = [(float(wp), float(wc)) for wp, wc in _species_list(species)]
    q = np.array([1.0])
    for _, wc in species:
        q = _poly.polymul(q, [-(wc**2), 1.0])
    sp = _poly.polymul([0.0, 1.0], q)
    for i, (wp, wc) in enumerate(species):
        others = np.array([1.0])
        for j, (_, wc_other) in enumerate(species):
            if j != i:
                others = _poly.polymul(others, [-(wc_other**2), 1.0])
        sp = _poly.polysub(sp, wp**2 * _poly.polymul([0.0, 1.0], others))
    pp = np.array([-sum(wp**2 for wp, _ in species), 1.0])
    # G = x²QRL = x (ωQ₊R)(ωQ₋L) with Q± = Π(ω ± Ω_s): polynomials in ω whose product is even
    g = _poly.polymul([0.0, 1.0], _poly.polymul(_resonance_free(species, 1), _resonance_free(species, -1))[::2])
    return q, sp, pp, g, species


def _resonance_free(species, sign):
    """ωQ±R (sign +1) or ωQ∓L (sign −1) as a polynomial in ω: ω Π(ω + sign Ω_t) − Σ_s ω_ps² Π_{t≠s}(ω + sign Ω_t)."""
    polynomial = np.array([0.0, 1.0])
    for _, wc in species:
        polynomial = _poly.polymul(polynomial, [sign * wc, 1.0])
    for i, (wp, _) in enumerate(species):
        others = np.array([1.0])
        for j, (_, wc) in enumerate(species):
            if j != i:
                others = _poly.polymul(others, [sign * wc, 1.0])
        polynomial = _poly.polysub(polynomial, wp**2 * others)
    return polynomial


def cold_plasma_waves(k, theta, species, c=1.0):
    """Compute all positive-frequency branches ω(k) of the cold magnetized plasma.

    For each k, the roots ω of Stix's dispersion relation A n⁴ − B n² + C = 0 with n = ck/ω
    (see :func:`refractive_index`), written as a polynomial in ω² of degree N + 3 for N species
    (free of the spurious roots at the cyclotron frequencies). All its roots are real and positive,
    so there are N + 3 branches: 4 for electrons alone (the two X-mode branches, the O mode and
    the whistler/electron-cyclotron wave), 5 for electrons and ions (adding the ion-cyclotron /
    shear Alfvén branch at low frequency). This is Struphy's ``ColdPlasma`` model (and
    ``ColdPlasma1D`` in ``struphy.dispersion_relations``).

    Parameters
    ----------
    k : float or array_like
        The wavenumber |k|.
    theta : float or array_like
        The angle between k and B₀, in radians.
    species : Species, (float, float) or sequence of these
        The species: plasma frequency and signed cyclotron frequency of each (scalars). A species
        with Ω_s = 0 (unmagnetized) turns one branch into ω = 0.
    c : float, optional
        The speed of light. Default: ``1.0``.

    Returns
    -------
    dict of str to complex or numpy.ndarray
        ``{"branch 1": ..., ..., "branch N+3": ...}``, numbered by ascending frequency at each k.
        Away from θ = 0 and π/2 the branches do not cross, so each is one continuous wave; at
        exactly θ = 0 or π/2 they can cross and then swap names at the crossing.

    See Also
    --------
    cutoffs : The frequencies of the branches at k = 0 (and the cold-plasma cutoffs).
    resonances : The frequencies they approach at large k.

    References
    ----------
    T. H. Stix, Waves in Plasmas (American Institute of Physics, 1992), ch. 1–2.

    Examples
    --------
    Electrons with ω_pe = |Ω_e| = 1, at 45° to B₀ and ck = 2:

    >>> w = cold_plasma_waves(2.0, np.pi / 4, Species(1.0, -1.0))
    >>> print({name: round(float(o.real), 4) for name, o in w.items()})
    {'branch 1': 0.4569, 'branch 2': 1.1994, 'branch 3': 2.1889, 'branch 4': 2.3583}
    """
    q, sp, pp, g, species = _cold_polynomials(species)
    k, theta = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(theta, dtype=float))
    degree = len(species) + 3

    def padded(p):
        return np.pad(p, (0, degree + 1 - len(p)))

    pp_q, pp_sp, pp_g = (padded(_poly.polymul(pp, p)) for p in (q, sp, g))
    sp, g = padded(sp), padded(g)
    n = (np.asarray(c, dtype=float) * k)[..., None] ** 2
    sin2, cos2 = (np.sin(theta) ** 2)[..., None], (np.cos(theta) ** 2)[..., None]
    a = sin2 * sp + cos2 * pp_q
    b = sin2 * g + (1 + cos2) * pp_sp
    coefficients = n**2 * a - n * b + pp_g
    roots = np.sqrt(np.maximum(_sorted_real_roots(coefficients), 0.0))
    roots = _polish_cold_roots(roots, n, sin2, cos2, species)
    return {f"branch {i + 1}": _complex(roots[..., i]) for i in range(degree)}


def _polish_cold_roots(omega, n, sin2, cos2, species):
    """Newton steps on Stix's A n⁴ − B n² + C = 0 (the expanded polynomial loses digits)."""

    def residual(w):
        s = stix(w, species)
        S, P, R, L = s["S"], s["P"], s["R"], s["L"]
        n2 = n / w**2
        return (S * sin2 + P * cos2) * n2**2 - (R * L * sin2 + P * S * (1 + cos2)) * n2 + P * R * L

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(3):
            h = 1e-7 * omega
            slope = (residual(omega + h) - residual(omega - h)) / (2 * h)
            step = residual(omega) / slope
            good = np.isfinite(step) & (np.abs(step) < 1e-3 * omega) & (omega > 0)
            omega = np.where(good, omega - np.where(good, step, 0.0), omega)
    return omega


def _positive_real_roots(ascending, tolerance=1e-9):
    roots = np.roots(np.asarray(ascending, dtype=float)[::-1])
    scale = np.max(np.abs(roots)) if roots.size else 1.0
    real = roots.real[np.abs(roots.imag) <= tolerance * max(scale, 1.0)]
    return np.sort(real[real > tolerance * max(scale, 1.0)])


def cutoffs(species):
    """Compute the cold-plasma cutoffs: the positive frequencies where R, L or P vanish.

    At a cutoff n² = 0 (k = 0), so these are the k → 0 limits of the branches of
    :func:`cold_plasma_waves`. For electrons alone,
    ω_R = ½(|Ω_e| + √(Ω_e² + 4ω_pe²)), ω_L = ½(−|Ω_e| + √(Ω_e² + 4ω_pe²)) and ω_P = ω_pe.

    Parameters
    ----------
    species : Species, (float, float) or sequence of these
        The species: plasma frequency and signed cyclotron frequency of each (scalars).

    Returns
    -------
    dict of str to numpy.ndarray
        ``{"R": ..., "L": ..., "P": ...}``, each the sorted positive (real) roots.

    References
    ----------
    T. H. Stix, Waves in Plasmas (American Institute of Physics, 1992), ch. 2.

    Examples
    --------
    >>> for name, w in cutoffs(Species(1.0, -1.0)).items():
    ...     print(name, np.round(w, 4))
    R [1.618]
    L [0.618]
    P [1.]
    """
    species = [(float(wp), float(wc)) for wp, wc in _species_list(species)]
    result = {}
    for name, sign in (("R", 1), ("L", -1)):
        result[name] = _positive_real_roots(_resonance_free(species, sign))
    result["P"] = np.array([np.sqrt(sum(wp**2 for wp, _ in species))])
    return result


def resonances(theta, species):
    """Compute the cold-plasma resonances at the angle θ: the positive frequencies where n² → ∞.

    The roots of A = S sin²θ + P cos²θ = 0, the large-k limits of the branches of
    :func:`cold_plasma_waves`. Across B₀ (θ = π/2) these are the hybrid resonances S = 0 (upper
    hybrid ω_UH = √(ω_pe² + Ω_e²) for electrons alone; for electrons and ions also the lower
    hybrid). Along B₀ (θ = 0 exactly) they are the cyclotron frequencies |Ω_s|, where R or L is
    infinite; for small θ > 0 there is one more close to ω = √(Σω_ps²), where P = 0.

    Parameters
    ----------
    theta : float
        The angle between k and B₀, in radians.
    species : Species, (float, float) or sequence of these
        The species: plasma frequency and signed cyclotron frequency of each (scalars).

    Returns
    -------
    numpy.ndarray
        The resonance frequencies, sorted.

    References
    ----------
    T. H. Stix, Waves in Plasmas (American Institute of Physics, 1992), ch. 2.

    Examples
    --------
    >>> print(np.round(resonances(np.pi / 2, Species(1.0, -1.0)), 4))   # upper hybrid, √2
    [1.4142]
    >>> print(np.round(resonances(0.0, electron_ion(1.0, 1.0, mass_ratio=100.0)), 4))
    [0.01 1.  ]
    """
    q, sp, pp, _, species = _cold_polynomials(species)
    sin2, cos2 = np.sin(theta) ** 2, np.cos(theta) ** 2
    if sin2 < 1e-30:
        return np.unique([abs(wc) for _, wc in species if wc != 0])
    # x Q A = Sp sin²θ + Pp Q cos²θ, a polynomial in x = ω²
    polynomial = _poly.polyadd(sin2 * sp, cos2 * _poly.polymul(pp, q))
    return np.sqrt(_positive_real_roots(polynomial))


def faraday_rotation(omega, length, species, c=1.0):
    """Compute the Faraday rotation angle of a linearly polarized wave traveling along B₀.

    The linear polarization is a sum of the R and L waves, which travel with n_R = √R and
    n_L = √L, so over a distance ``length`` the plane of polarization turns by

    ψ = ω (n_L − n_R) length/(2c),

    positive in the sense of electron gyration (counter-clockwise looking against B₀). At high
    frequency, ψ ≈ Σ_s (−Ω_s) ω_ps² length/(2cω²), for electrons ω_pe² |Ω_e| length/(2cω²).

    Parameters
    ----------
    omega : float or array_like
        The wave frequency, above the cutoffs of both the R and the L wave.
    length : float or array_like
        The distance traveled along B₀.
    species : Species, (float, float) or sequence of these
        The species: plasma frequency and signed cyclotron frequency of each.
    c : float or array_like, optional
        The speed of light. Default: ``1.0``.

    Returns
    -------
    float or numpy.ndarray
        ψ in radians; ``nan`` where the R or the L wave does not propagate (n² < 0).

    References
    ----------
    F. F. Chen, Introduction to Plasma Physics and Controlled Fusion, 3rd ed. (Springer, 2016),
    ch. 4.
    T. H. Stix, Waves in Plasmas (American Institute of Physics, 1992).

    Examples
    --------
    >>> print(round(float(faraday_rotation(10.0, 100.0, Species(1.0, -1.0))), 4))
    0.5076
    """
    s = stix(np.asarray(omega, dtype=float), species)
    with np.errstate(invalid="ignore"):
        n_r, n_l = np.sqrt(s["R"]), np.sqrt(s["L"])
    return np.asarray(np.asarray(omega) * (n_l - n_r) * np.asarray(length) / (2 * np.asarray(c)))[()]


def group_velocity(omega_of_k, k, step=None):
    """Compute the group velocity dω/dk of a dispersion relation by central differences.

    Parameters
    ----------
    omega_of_k : callable
        ω(k), returning an array or a dict of branch names to arrays (like the functions of this
        module).
    k : float or array_like
        The wavenumbers.
    step : float, optional
        The finite-difference step. Default: 10⁻⁶ max(1, |k|).

    Returns
    -------
    complex, numpy.ndarray or dict
        dω/dk (complex), a dict of branch names to it if ``omega_of_k`` returns a dict.

    References
    ----------
    T. H. Stix, Waves in Plasmas (American Institute of Physics, 1992).

    Examples
    --------
    >>> v = group_velocity(lambda k: plasma_light_wave(k, plasma_frequency=1.0), 1.0)
    >>> print(round(float(v.real), 6))    # c²k/ω = 1/√2
    0.707107
    """
    k = np.asarray(k, dtype=float)
    h = np.maximum(1.0, np.abs(k)) * 1e-6 if step is None else np.asarray(step, dtype=float)
    upper, lower = omega_of_k(k + h), omega_of_k(k - h)

    def derivative(a, b):
        return _complex((np.asarray(a) - np.asarray(b)) / (2 * h))

    if isinstance(upper, dict):
        return {name: derivative(upper[name], lower[name]) for name in upper}
    return derivative(upper, lower)


# ----------------------------------------------------------------------------------------------
# Cavities
# ----------------------------------------------------------------------------------------------


def cavity_modes(lengths, c=1.0, max_index=6, max_frequency=None):
    """List the resonant modes of a rectangular cavity with perfectly conducting walls.

    ω = cπ √((l/a)² + (m/b)² + (n/d)²) for the side lengths (a, b, d), counted with their
    polarizations relative to the last axis:

    * TM (E_z ≠ 0): l, m ≥ 1 and n ≥ 0,
    * TE (H_z ≠ 0): n ≥ 1 and (l, m) ≠ (0, 0),

    so a mode with all indices ≥ 1 comes twice, one with one zero index once, and none with two
    zero indices. For two lengths (a, b), a two-dimensional cavity with the fields uniform along
    z: TM (E_z) modes with l, m ≥ 1 and TE (H_z) modes with (l, m) ≠ (0, 0).

    Parameters
    ----------
    lengths : sequence of float
        The side lengths (a, b, d), or (a, b) for a two-dimensional cavity.
    c : float, optional
        The speed of light in the cavity. Default: ``1.0``.
    max_index : int, optional
        The largest index l, m, n listed. Default: ``6``.
    max_frequency : float, optional
        List only modes with ω ≤ ``max_frequency`` (make sure ``max_index`` reaches it).
        Default: all up to ``max_index``.

    Returns
    -------
    dict
        ``{"omega": ..., "indices": ..., "kind": ...}``: the (real) frequencies sorted ascending,
        an integer array of shape (modes, len(lengths)) with (l, m, n), and the array of ``"TE"``
        or ``"TM"``.

    Raises
    ------
    ValueError
        If ``lengths`` does not have 2 or 3 entries.

    References
    ----------
    J. D. Jackson, Classical Electrodynamics, 3rd ed. (Wiley, 1999), ch. 8.
    D. M. Pozar, Microwave Engineering, 4th ed. (Wiley, 2012), ch. 6.

    Examples
    --------
    >>> modes = cavity_modes((1.0, 1.0, 1.0), max_index=2)
    >>> for w, idx, kind in list(zip(modes["omega"], modes["indices"], modes["kind"]))[:4]:
    ...     print(round(float(w / np.pi), 4), idx, kind)
    1.4142 [0 1 1] TE
    1.4142 [1 0 1] TE
    1.4142 [1 1 0] TM
    1.7321 [1 1 1] TE
    """
    lengths = np.asarray(lengths, dtype=float)
    if lengths.shape not in ((2,), (3,)):
        raise ValueError(f"lengths must have 2 or 3 entries; got {lengths.shape}")
    grid = np.stack(np.meshgrid(*[np.arange(max_index + 1)] * len(lengths), indexing="ij"), axis=-1)
    indices = grid.reshape(-1, len(lengths))
    if len(lengths) == 3:
        tm = (indices[:, 0] >= 1) & (indices[:, 1] >= 1)
        te = (indices[:, 2] >= 1) & ((indices[:, 0] >= 1) | (indices[:, 1] >= 1))
    else:
        tm = (indices[:, 0] >= 1) & (indices[:, 1] >= 1)
        te = (indices[:, 0] >= 1) | (indices[:, 1] >= 1)
    indices = np.concatenate([indices[te], indices[tm]])
    kind = np.array(["TE"] * int(te.sum()) + ["TM"] * int(tm.sum()))
    omega = np.pi * c * np.sqrt(np.sum((indices / lengths) ** 2, axis=1))
    if max_frequency is not None:
        keep = omega <= max_frequency * (1 + 1e-12)
        omega, indices, kind = omega[keep], indices[keep], kind[keep]
    order = np.lexsort((kind, *indices.T[::-1], np.round(omega, 12)))
    return {"omega": omega[order], "indices": indices[order], "kind": kind[order]}


# ----------------------------------------------------------------------------------------------
# Drift waves
# ----------------------------------------------------------------------------------------------


def drift_wave(ky, kx=0.0, diamagnetic_speed=1.0, rho_s=1.0):
    """Compute the frequency of the electron drift wave with adiabatic electrons.

    ω = ω*/(1 + k⊥²ρ_s²),   ω* = k_y v*,   k⊥² = k_x² + k_y²,

    the Hasegawa–Mima drift wave: v* = T_e/(eB₀L_n) = ρ_s c_s/L_n is the electron diamagnetic
    drift speed, positive along y (the electron diamagnetic direction, with x down the density
    gradient: n₀ ∝ exp(−x/L_n)). In the usual normalization (lengths in ρ_s, time in L_n/c_s)
    v* = ρ_s = 1 and ω = k_y/(1 + k⊥²). In the Hasegawa–Wakatani normalization of
    :func:`hasegawa_wakatani` v* = κ.

    Parameters
    ----------
    ky : float or array_like
        The wavenumber across B₀ and the density gradient.
    kx : float or array_like, optional
        The wavenumber along the density gradient. Default: ``0.0``.
    diamagnetic_speed : float or array_like, optional
        v*. Default: ``1.0``.
    rho_s : float or array_like, optional
        The ion sound radius ρ_s = c_s/Ω_i. Default: ``1.0``.

    Returns
    -------
    complex or numpy.ndarray
        ω, complex (real-valued).

    References
    ----------
    A. Hasegawa and K. Mima, "Pseudo-three-dimensional turbulence in magnetized nonuniform
    plasma", Phys. Fluids 21, 87 (1978).
    F. F. Chen, Introduction to Plasma Physics and Controlled Fusion, 3rd ed. (Springer, 2016),
    ch. 6.

    Examples
    --------
    >>> print(round(float(drift_wave(1.0).real), 4))
    0.5
    """
    ky, kx = np.asarray(ky, dtype=float), np.asarray(kx, dtype=float)
    rho = np.asarray(rho_s, dtype=float)
    return _complex(ky * np.asarray(diamagnetic_speed) / (1 + (kx**2 + ky**2) * rho**2))


def hasegawa_wakatani(ky, kx=0.0, adiabaticity=1.0, gradient=1.0, viscosity=0.0):
    """Compute the two linear modes of the Hasegawa–Wakatani equations.

    The (modified) Hasegawa–Wakatani equations as in Struphy's ``HasegawaWakatani`` model,

    ∂ζ/∂t + [φ, ζ] = α(φ − n) + ν∇²ζ,
    ∂n/∂t + [φ, n] = α(φ − n) − κ ∂φ/∂y + ν∇²n,   ζ = ∇²φ,

    in lengths of ρ_s and times of 1/Ω_i (φ in T_e/e and n in n₀, both scaled by L_n/ρ_s), with the adiabaticity α (parallel electron conductivity; α → ∞ gives adiabatic electrons) and
    the density gradient κ = ρ_s/L_n. Linearized with exp(i(k_x x + k_y y − ωt)), k² = k_x² + k_y²:

    ω² + i ω [α(1 + k²)/k²] − i α κ k_y/k² = 0   (for ν = 0; ν shifts both roots by −iνk²).

    One root is the drift wave, unstable for every k_y ≠ 0 at finite α (resistive drift-wave
    instability), with ω → κk_y/(1 + k²) (:func:`drift_wave`) and growth → 0 as α → ∞; the other
    is damped.

    Parameters
    ----------
    ky : float or array_like
        The wavenumber along y (across the gradient and B₀).
    kx : float or array_like, optional
        The wavenumber along x (the gradient). Default: ``0.0``.
    adiabaticity : float or array_like, optional
        α. Default: ``1.0``.
    gradient : float or array_like, optional
        κ. Default: ``1.0``.
    viscosity : float or array_like, optional
        ν (Struphy's ν∇² dissipation on both fields). Default: ``0.0``.

    Returns
    -------
    dict of str to complex or numpy.ndarray
        ``{"drift wave": ..., "damped": ...}``: the root with the larger imaginary part (the
        unstable drift wave) and the other one.

    References
    ----------
    A. Hasegawa and M. Wakatani, "Plasma edge turbulence", Phys. Rev. Lett. 50, 682 (1983).
    S. J. Camargo, D. Biskamp and B. D. Scott, "Resistive drift-wave turbulence",
    Phys. Plasmas 2, 48 (1995).

    Examples
    --------
    >>> w = hasegawa_wakatani(1.0, adiabaticity=1.0, gradient=1.0)["drift wave"]
    >>> print(round(float(w.real), 4), round(float(w.imag), 4))
    0.4551 0.0987
    """
    ky, kx, alpha, kappa, nu = np.broadcast_arrays(
        *(np.asarray(v, dtype=float) for v in (ky, kx, adiabaticity, gradient, viscosity))
    )
    k2 = kx**2 + ky**2
    with np.errstate(divide="ignore", invalid="ignore"):
        b = 1j * alpha * (1 + k2) / k2
        c = -1j * alpha * kappa * ky / k2
    root = np.sqrt(b**2 - 4 * c)
    first, second = (-b + root) / 2 - 1j * nu * k2, (-b - root) / 2 - 1j * nu * k2
    swap = second.imag > first.imag
    return {"drift wave": _complex(np.where(swap, second, first)), "damped": _complex(np.where(swap, first, second))}


# ----------------------------------------------------------------------------------------------
# Continua and the TAE gap
# ----------------------------------------------------------------------------------------------


def parallel_wavenumber(r, m, n, q, major_radius=1.0):
    """Compute the parallel wavenumber k∥ = (n + m/q(r))/R₀ of a Fourier mode in a cylinder.

    For the mode exp(i(mθ + nφ)) with φ = z/R₀ in a periodic cylinder (or a large-aspect-ratio
    tokamak), in Struphy's sign convention (``MhdContinousSpectraCylinder``,
    ``MhdContinousSpectraShearedSlab``). It vanishes at the rational surface q = −m/n; with the
    other common convention k∥ = (n − m/q)/R₀, flip the sign of m.

    Parameters
    ----------
    r : float or array_like
        The minor radius.
    m : int or array_like
        The poloidal mode number.
    n : int or array_like
        The toroidal mode number.
    q : float, array_like or callable
        The safety factor: a number, an array (of the shape of ``r``) or a function of r.
    major_radius : float, optional
        R₀. Default: ``1.0``.

    Returns
    -------
    float or numpy.ndarray
        k∥.

    References
    ----------
    J. P. Goedbloed and S. Poedts, Principles of Magnetohydrodynamics (Cambridge University Press,
    2004), ch. 9.

    Examples
    --------
    >>> print(float(parallel_wavenumber(0.5, m=2, n=-1, q=2.0, major_radius=3.0)))
    0.0
    """
    r = np.asarray(r, dtype=float)
    return np.asarray((np.asarray(n) + np.asarray(m) / _of_r(q, r)) / major_radius)[()]


def alfven_continuum(r, m, n, q, major_radius=1.0, alfven_speed=1.0):
    """Compute the shear Alfvén continuum ω(r) = |k∥(r)| v_A(r) of a cylinder.

    With k∥ = (n + m/q)/R₀ as in :func:`parallel_wavenumber` (Struphy's convention: this is the
    ``"shear Alfvén"`` branch of ``MhdContinousSpectraCylinder`` with v_A² = B₀z²/n₀).

    Parameters
    ----------
    r : float or array_like
        The minor radius.
    m : int or array_like
        The poloidal mode number.
    n : int or array_like
        The toroidal mode number.
    q : float, array_like or callable
        The safety factor: a number, an array or a function of r.
    major_radius : float, optional
        R₀. Default: ``1.0``.
    alfven_speed : float, array_like or callable, optional
        v_A: a number, an array or a function of r. Default: ``1.0``.

    Returns
    -------
    complex or numpy.ndarray
        ω(r), complex (real-valued).

    References
    ----------
    J. P. Goedbloed and S. Poedts, Principles of Magnetohydrodynamics (Cambridge University Press,
    2004), ch. 9.
    J. P. Freidberg, Ideal MHD (Cambridge University Press, 2014).

    Examples
    --------
    >>> r = np.array([0.0, 0.5, 1.0])
    >>> w = alfven_continuum(r, m=2, n=-1, q=lambda r: 1 + r**2, major_radius=3.0)
    >>> print(np.round(w.real, 4))
    [0.3333 0.2    0.    ]
    """
    r = np.asarray(r, dtype=float)
    return _complex(np.abs(parallel_wavenumber(r, m, n, q, major_radius)) * _of_r(alfven_speed, r))


def slow_continuum(r, m, n, q, major_radius=1.0, alfven_speed=1.0, sound_speed=0.5):
    """Compute the slow (cusp) continuum ω(r) = |k∥| c_s v_A/√(c_s² + v_A²) of a cylinder.

    With k∥ = (n + m/q)/R₀ as in :func:`parallel_wavenumber` (Struphy's ``"slow sound"`` branch of
    ``MhdContinousSpectraCylinder``, for B₀θ ≪ B₀z).

    Parameters
    ----------
    r : float or array_like
        The minor radius.
    m : int or array_like
        The poloidal mode number.
    n : int or array_like
        The toroidal mode number.
    q : float, array_like or callable
        The safety factor: a number, an array or a function of r.
    major_radius : float, optional
        R₀. Default: ``1.0``.
    alfven_speed : float, array_like or callable, optional
        v_A: a number, an array or a function of r. Default: ``1.0``.
    sound_speed : float, array_like or callable, optional
        c_s = √(γp₀/ρ₀): a number, an array or a function of r. Default: ``0.5``.

    Returns
    -------
    complex or numpy.ndarray
        ω(r), complex (real-valued).

    References
    ----------
    J. P. Goedbloed and S. Poedts, Principles of Magnetohydrodynamics (Cambridge University Press,
    2004), ch. 9.

    Examples
    --------
    >>> w = slow_continuum(0.0, m=1, n=0, q=1.0, alfven_speed=1.0, sound_speed=1.0)
    >>> print(round(float(w.real), 4))     # 1/√2
    0.7071
    """
    r = np.asarray(r, dtype=float)
    va, cs = _of_r(alfven_speed, r), _of_r(sound_speed, r)
    cusp = cs * va / np.sqrt(cs**2 + va**2)
    return _complex(np.abs(parallel_wavenumber(r, m, n, q, major_radius)) * cusp)


def tae_frequency(q, major_radius=1.0, alfven_speed=1.0):
    """Compute the frequency ω_TAE = v_A/(2|q|R₀) at the center of the toroidal Alfvén gap.

    Where the continua of the poloidal harmonics m and m + 1 would cross, |k∥| = 1/(2|q|R₀); the
    toroidal coupling opens a gap there, in which the TAE lives.

    Parameters
    ----------
    q : float or array_like
        The safety factor at the gap, q = |m + ½|/|n|.
    major_radius : float or array_like, optional
        R₀. Default: ``1.0``.
    alfven_speed : float or array_like, optional
        v_A at the gap. Default: ``1.0``.

    Returns
    -------
    float or numpy.ndarray
        ω_TAE (real).

    References
    ----------
    C. Z. Cheng, L. Chen and M. S. Chance, "High-n ideal and resistive shear Alfvén waves in
    tokamaks", Ann. Phys. 161, 21 (1985).
    W. W. Heidbrink, "Basic physics of Alfvén instabilities driven by energetic particles in
    toroidally confined plasmas", Phys. Plasmas 15, 055501 (2008).

    Examples
    --------
    >>> print(round(float(tae_frequency(1.5, major_radius=3.0)), 4))
    0.1111
    """
    return np.asarray(np.asarray(alfven_speed, dtype=float) / (2 * np.abs(q) * np.asarray(major_radius)))[()]


