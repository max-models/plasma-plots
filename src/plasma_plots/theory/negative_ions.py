"""Negative hydrogen ions (H⁻): destruction processes, electronegative plasma and beam survival, in SI units.

Densities are in m⁻³, temperatures and energies in eV (k_B T / e), cross sections in m², rate
coefficients in m³/s, frequencies in 1/s and speeds in m/s, as in
:mod:`~plasma_plots.theory.parameters`, which this module reuses together with
:mod:`~plasma_plots.theory.sheaths` and :mod:`~plasma_plots.theory.beams`.

Contents
--------
* **Cross sections** (fits of Janev, Reiter and Samm): :func:`electron_detachment_cross_section`
  (e + H⁻), :func:`mutual_neutralization_cross_section` (H⁺ + H⁻),
  :func:`hydrogen_detachment_cross_section` (H + H⁻) and
  :func:`h2_detachment_cross_section` (H₂ + H⁻).
* **Rates**: :func:`maxwellian_rate_coefficient` (any cross section averaged over Maxwellians),
  :func:`electron_detachment_rate_coefficient`, :func:`mutual_neutralization_rate_coefficient` and
  the H⁻ loss frequencies of a plasma, :func:`destruction_frequencies`.
* **Electronegative plasma**: :func:`debye_length_electronegative`,
  :func:`sheath_edge_electronegativity`, :func:`bohm_speed_electronegative` and
  :func:`ion_flux_electronegative`.
* **Beams and extraction**: :func:`beam_survival_fraction` (stripping on H₂ and H, with
  :func:`~plasma_plots.theory.beams.neutral_beam_attenuation`) and
  :func:`surface_production_current`.

Conventions
-----------
* **Cross-section fits** are evaluated exactly as printed in the source, in the stated energy
  variable (electron energy, relative collision energy, or laboratory energy of the H⁻). Outside the
  stated fit range they are extrapolations: the module does not clip, check the ranges in each
  docstring.
* **Rates** are Maxwellian averages <σ v> of the fits. Species with different temperatures are
  handled through the temperature of the relative motion, T_rel = (m_b T_a + m_a T_b) / (m_a + m_b).
* **Electronegativity** is α = n₋ / n_e. Quasi-neutrality of the positive ions, n₊ = n_e (1 + α),
  is assumed, and ions carry one elementary charge.
* **Boltzmann negative ions**: the electron and H⁻ densities follow T_e ∇ln n_e = T₋ ∇ln n₋, so
  γ = T_e / T₋ enters the Bohm speed. This fails at high pressure, where the H⁻ are collisional.

References
----------
R. K. Janev, D. Reiter and U. Samm, Collision Processes in Low-Temperature Hydrogen Plasmas,
Berichte des Forschungszentrums Jülich 4105 (2003), section 3 (H⁻ ions) and section 6.1 (H⁻ + H₂),
https://eirene.de/Documentation/report_4105.pdf.

E. Kemaneci et al., A global model of cylindrical and coaxial surface-wave discharges,
arXiv:1612.07268, eqs. (10) and (11).

N. St. J. Braithwaite and J. E. Allen, J. Phys. D 21, 1733 (1988).

Examples
--------
The Maxwellian rate coefficients of the two main plasma losses of H⁻ at T_e = 1 eV, and the
fraction of a 50 keV H⁻ beam that survives 0.5 m of 0.01 Pa H₂ gas:

>>> print(f"{electron_detachment_rate_coefficient(1.0):.3e} m^3/s")
1.472e-14 m^3/s
>>> print(f"{mutual_neutralization_rate_coefficient(1.0):.3e} m^3/s")
6.092e-14 m^3/s
>>> print(f"{beam_survival_fraction(50e3, h2_density=2.4e18, length=0.5):.4f}")
0.9450
"""

from __future__ import annotations

import numpy as np

from plasma_plots.theory.beams import neutral_beam_attenuation
from plasma_plots.theory.parameters import (
    debye_length,
    electron_mass,
    elementary_charge,
    proton_mass,
)
from plasma_plots.theory.sheaths import bohm_speed

_CM2 = 1e-4  # m² per cm²
#: Binding energy of the extra electron of H⁻, in eV (Janev et al., section 3.1.1).
DETACHMENT_THRESHOLD = 0.754

# Table 10 of Janev et al.: a_0 ... a_8 of ln σ[cm²] = Σ a_j (ln E[eV])^j, H⁺ + H⁻ → H(n) + H.
_MN_COEFFICIENTS = {
    2: (
        -3.49880888e1,
        2.15245051e-1,
        -2.35628664e-2,
        5.49471553e-2,
        5.37932888e-3,
        -6.05507021e-3,
        9.99168329e-4,
        -6.63625564e-5,
        1.61228385e-6,
    ),
    3: (
        -3.11479336e1,
        -7.73020527e-1,
        5.49204378e-2,
        -2.73324984e-3,
        -1.22831288e-3,
        4.35049828e-4,
        -6.21659501e-5,
        4.12046807e-6,
        -1.039784996e-7,
    ),
}

# Table 11 of Janev et al.: the same polynomial for H⁻ + H → H + H + e and → H₂ + e.
_HH_COEFFICIENTS = {
    "detachment": (
        -3.61799082e1,
        1.16615172,
        -1.41928602e-1,
        -1.11195959e-2,
        -1.72505995e-3,
        1.59040356e-3,
        -2.53196144e-4,
        1.66978235e-5,
        -4.09725797e-7,
    ),
    "associative": (
        -3.44152907e1,
        -3.39348209e-1,
        5.66591705e-2,
        -9.05150459e-3,
        7.66060418e-4,
        -4.27126462e-5,
        -1.57273749e-7,
        2.57607677e-7,
        -1.20071919e-8,
    ),
}


def _out(value):
    return np.asarray(value, dtype=float)[()]


def _array(value):
    return np.asarray(value, dtype=float)


def _polynomial_cross_section(energy, coefficients):
    """exp(Σ a_j (ln E)^j) in m², for E in eV (Horner scheme)."""
    log_e = np.log(_array(energy))
    total = np.zeros_like(log_e)
    for a in reversed(coefficients):
        total = total * log_e + a
    return np.exp(total) * _CM2


# ---------------------------------------------------------------------------------------------
# Cross sections
# ---------------------------------------------------------------------------------------------
def electron_detachment_cross_section(energy):
    """Compute the cross section of electron-impact detachment, e + H⁻ → e + H + e.

    The fit σ = 2.06/E · ln(e + 2.335e-4 E) · exp(−13.75/E^0.868) · [1 − (0.754/E)²] × 1e-13 cm²
    of Janev et al., eq. (58), with E the electron energy in eV (the figure of the source covers
    1 eV to 10 keV). Below the threshold of 0.754 eV, the binding energy of H⁻, the cross section
    is zero.

    Parameters
    ----------
    energy : float or array_like
        Electron energy E, in eV.

    Returns
    -------
    float or numpy.ndarray
        σ, in m².

    References
    ----------
    Janev, Reiter and Samm, Jül-4105, section 3.1.1, eq. (58).

    Examples
    --------
    The source quotes σ ≈ 2.3e-15 cm² at 65 eV:

    >>> print(f"{electron_detachment_cross_section(65.0) * 1e4:.3e} cm^2")
    2.208e-15 cm^2
    """
    energy = _array(energy)
    safe = np.where(energy > DETACHMENT_THRESHOLD, energy, 1.0)
    sigma = (
        2.06
        / safe
        * np.log(np.e + 2.335e-4 * safe)
        * np.exp(-13.75 / safe**0.868)
        * (1.0 - (DETACHMENT_THRESHOLD / safe) ** 2)
        * 1e-13
        * _CM2
    )
    return _out(np.where(energy > DETACHMENT_THRESHOLD, sigma, 0.0))


def mutual_neutralization_cross_section(energy, final_level=None):
    """Compute the cross section of mutual neutralization, H⁺ + H⁻ → H(n) + H(1s), n = 2, 3.

    The polynomial fits ln σ = Σ_{j=0..8} a_j (ln E)^j of Janev et al., eq. (63), with the
    coefficients of their table 10, in cm² and eV; the source gives them for the relative collision
    energy 0.1 eV to 20 keV. The n = 1 and n ≥ 4 channels are not populated.

    Parameters
    ----------
    energy : float or array_like
        Relative collision energy E, in eV (the centre-of-mass energy).
    final_level : {2, 3}, optional
        Return only the capture into this level of the H atom. Default: the sum of both.

    Returns
    -------
    float or numpy.ndarray
        σ, in m².

    References
    ----------
    Janev, Reiter and Samm, Jül-4105, section 3.2.1, eq. (63) and table 10.

    Examples
    --------
    >>> print(f"{mutual_neutralization_cross_section(1.0) * 1e4:.4e} cm^2")
    3.0329e-14 cm^2
    """
    if final_level is None:
        return _out(
            _polynomial_cross_section(energy, _MN_COEFFICIENTS[2])
            + _polynomial_cross_section(energy, _MN_COEFFICIENTS[3])
        )
    if final_level not in _MN_COEFFICIENTS:
        raise ValueError(f"final_level must be 2 or 3, got {final_level!r}")
    return _out(_polynomial_cross_section(energy, _MN_COEFFICIENTS[final_level]))


def hydrogen_detachment_cross_section(energy, channel="detachment"):
    """Compute the cross section of H⁻ + H(1s) detachment, to H + H + e or to H₂ + e.

    The polynomial fit ln σ = Σ_{j=0..8} a_j (ln E)^j of Janev et al., eq. (71), with the
    coefficients of their table 11, valid for the relative collision energy 0.1 eV to 20 keV.

    Parameters
    ----------
    energy : float or array_like
        Relative collision energy E, in eV (the centre-of-mass energy).
    channel : {"detachment", "associative"}, optional
        ``"detachment"``: H⁻ + H → H + H + e. ``"associative"``: H⁻ + H → H₂ + e. Default:
        ``"detachment"``.

    Returns
    -------
    float or numpy.ndarray
        σ, in m².

    References
    ----------
    Janev, Reiter and Samm, Jül-4105, section 3.3.2, eq. (71) and table 11.

    Examples
    --------
    >>> print(f"{hydrogen_detachment_cross_section(10.0) * 1e4:.4e} cm^2")
    1.1947e-15 cm^2
    """
    if channel not in _HH_COEFFICIENTS:
        raise ValueError(f"channel must be one of {tuple(_HH_COEFFICIENTS)}, got {channel!r}")
    return _out(_polynomial_cross_section(energy, _HH_COEFFICIENTS[channel]))


def h2_detachment_cross_section(energy):
    """Compute the cross section of H⁻ + H₂(v = 0) → H + H₂ + e, the stripping of an H⁻ beam on H₂.

    The sum σ = σ< + σ> of Janev et al., eqs. (159a–c), with the laboratory energy E of the H⁻:
    σ< = 81.10e-16 cm² / (E^0.63 [1 + 2.03e-6 E^1.95]) · {1 − exp[−0.057 (E/2.18 − 1)^0.94]} and
    σ> = 1.22e3 e-16 cm² / (E^0.5 [1 + 6.91e-4 E^0.40]) · exp(−125.0/E^0.663). The source states
    "E in keV" below eq. (159c), but the threshold 2.18 in (159b) is the 2.18 eV endothermicity it
    quotes, and only E in eV reproduces its figure 34 and its stated maximum of about 1e-15 cm²
    near 2 keV, so E is in eV here. The source fits measurements from threshold to the MeV range.
    σ< is zero below the 2.18 eV threshold, where its printed form is undefined.

    Parameters
    ----------
    energy : float or array_like
        Laboratory energy of the H⁻, in eV.

    Returns
    -------
    float or numpy.ndarray
        σ, in m².

    References
    ----------
    Janev, Reiter and Samm, Jül-4105, section 6.1.1, eqs. (159a–c) and figure 34.

    Examples
    --------
    The maximum of about 1.2e-15 cm² near 2 keV:

    >>> print(f"{h2_detachment_cross_section(2e3) * 1e4:.3e} cm^2")
    1.207e-15 cm^2
    """
    energy = _array(energy)
    above = energy > 2.18
    lower = (
        81.10e-16
        / (energy**0.63 * (1.0 + 2.03e-6 * energy**1.95))
        * (1.0 - np.exp(-0.057 * np.where(above, energy / 2.18 - 1.0, 0.0) ** 0.94))
    )
    upper = 1.22e3 * 1e-16 / (energy**0.5 * (1.0 + 6.91e-4 * energy**0.40)) * np.exp(-125.0 / energy**0.663)
    return _out((np.where(above, lower, 0.0) + upper) * _CM2)


# ---------------------------------------------------------------------------------------------
# Rates
# ---------------------------------------------------------------------------------------------
def maxwellian_rate_coefficient(
    cross_section,
    temperature,
    mass_a=electron_mass,
    mass_b=proton_mass,
    temperature_b=None,
    energy_range=None,
    points=2000,
):
    """Compute the rate coefficient <σ v> of a cross section σ(E) averaged over two Maxwellians.

    With the relative motion Maxwellian at T_rel = (m_b T_a + m_a T_b)/(m_a + m_b) and the
    reduced mass μ, <σ v> = √(8/(π μ)) T_rel^(−3/2) ∫ σ(E) E exp(−E/T_rel) dE (E and T in energy
    units, E the relative energy ½ μ v²), integrated here with the trapezoidal rule on a
    logarithmic energy grid.

    Parameters
    ----------
    cross_section : callable
        σ(E) in m², taking the relative energy E in eV as a numpy array.
    temperature : float or array_like
        Temperature T_a of the first species, in eV.
    mass_a, mass_b : float, optional
        Masses in kg. Defaults: an electron and a proton (the reduced mass is m_e to within
        m_e/m_H⁻ for e + H⁻); pass two proton masses for H⁺ + H⁻.
    temperature_b : float or array_like, optional
        Temperature of the second species, in eV. Default: ``temperature``.
    energy_range : tuple of float, optional
        Lowest and highest relative energy of the integral, in eV; the part of the Maxwellian
        outside is dropped. Default: 1e-4 to 60 times T_rel.
    points : int, optional
        Number of grid points. Default: ``2000``.

    Returns
    -------
    float or numpy.ndarray
        <σ v>, in m³/s.

    References
    ----------
    Standard Maxwellian average of a relative-motion cross section, e.g. Janev et al., Jül-4105
    (the cross sections themselves); the effective temperature of two Maxwellians is the one
    of their relative velocity distribution.

    Examples
    --------
    A constant σ = 1e-19 m² between electrons and protons at 1 eV:

    >>> print(
    ...     f"{maxwellian_rate_coefficient(lambda e: 1e-19 + 0 * e, 1.0):.4e} m^3/s"
    ... )
    6.6942e-14 m^3/s
    """
    t_a = _array(temperature)
    t_b = t_a if temperature_b is None else _array(temperature_b)
    t_rel = (mass_b * t_a + mass_a * t_b) / (mass_a + mass_b)
    mu = mass_a * mass_b / (mass_a + mass_b)
    t_rel = np.atleast_1d(t_rel)
    if energy_range is None:
        lo, hi = 1e-4 * t_rel, 60.0 * t_rel
    else:
        lo, hi = np.full_like(t_rel, energy_range[0]), np.full_like(t_rel, energy_range[1])
    u = np.linspace(0.0, 1.0, points)
    energy = np.exp(np.log(lo)[..., None] + (np.log(hi) - np.log(lo))[..., None] * u)
    # ∫ f(E) E dE = ∫ f(E) E² d ln E
    integrand = cross_section(energy) * np.exp(-energy / t_rel[..., None]) * energy**2
    integral = np.trapezoid(integrand, np.log(energy), axis=-1)
    rate = np.sqrt(8.0 * elementary_charge / (np.pi * mu)) * t_rel**-1.5 * integral
    return _out(rate.reshape(np.shape(t_a)) if np.ndim(t_a) else rate[0])


def electron_detachment_rate_coefficient(electron_temperature, negative_ion_temperature=None):
    """Compute the rate coefficient of e + H⁻ → e + H + e for Maxwellian electrons.

    <σ v> of :func:`electron_detachment_cross_section`: the relative energy is the electron
    energy to within m_e/m_H⁻, so the H⁻ temperature hardly matters.

    Parameters
    ----------
    electron_temperature : float or array_like
        T_e, in eV.
    negative_ion_temperature : float or array_like, optional
        H⁻ temperature, in eV. Default: ``electron_temperature``.

    Returns
    -------
    float or numpy.ndarray
        <σ v>, in m³/s. The threshold is 0.754 eV, so the rate falls steeply below T_e ≈ 0.3 eV.

    References
    ----------
    Maxwellian average of Janev et al., Jül-4105, eq. (58).

    Examples
    --------
    >>> print(f"{electron_detachment_rate_coefficient(1.0):.3e} m^3/s")
    1.472e-14 m^3/s
    """
    return maxwellian_rate_coefficient(
        electron_detachment_cross_section,
        electron_temperature,
        mass_a=electron_mass,
        mass_b=proton_mass,
        temperature_b=negative_ion_temperature,
    )


def mutual_neutralization_rate_coefficient(
    positive_ion_temperature, negative_ion_temperature=None, energy_range=(0.1, 2e4)
):
    """Compute the thermal rate coefficient of H⁺ + H⁻ → H(n) + H for Maxwellian ions.

    <σ v> of :func:`mutual_neutralization_cross_section`, with the reduced mass m_p / 2 and the
    relative temperature (T₊ + T₋)/2 (equal masses).

    Parameters
    ----------
    positive_ion_temperature : float or array_like
        H⁺ temperature, in eV.
    negative_ion_temperature : float or array_like, optional
        H⁻ temperature, in eV. Default: ``positive_ion_temperature``.
    energy_range : tuple of float, optional
        Relative energies of the integral, in eV. Default: ``(0.1, 2e4)``, the range of validity of
        the fit; the part of the Maxwellian below 0.1 eV is dropped, which underestimates the rate
        for T below about 0.3 eV.

    Returns
    -------
    float or numpy.ndarray
        <σ v>, in m³/s.

    References
    ----------
    Maxwellian average of Janev et al., Jül-4105, eq. (63) and table 10.

    Examples
    --------
    >>> print(f"{mutual_neutralization_rate_coefficient(1.0):.3e} m^3/s")
    6.092e-14 m^3/s
    """
    return maxwellian_rate_coefficient(
        mutual_neutralization_cross_section,
        positive_ion_temperature,
        mass_a=proton_mass,
        mass_b=proton_mass,
        temperature_b=negative_ion_temperature,
        energy_range=energy_range,
    )


def destruction_frequencies(
    electron_density,
    electron_temperature,
    positive_ion_density=None,
    ion_temperature=1.0,
    atom_density=0.0,
    atom_temperature=0.1,
):
    """Compute the loss frequencies ν = n <σ v> of an H⁻ ion in a hydrogen plasma, per channel.

    Channels: electron detachment (e + H⁻), mutual neutralization (H⁺ + H⁻) and the detachment of
    H⁻ by H atoms (both the H + H + e and the H₂ + e channel of eq. 71 together). The loss by H₂
    gas is not included: it needs the energy of a beam (:func:`h2_detachment_cross_section`).

    Parameters
    ----------
    electron_density : float or array_like
        n_e, in m⁻³.
    electron_temperature : float or array_like
        T_e, in eV.
    positive_ion_density : float or array_like, optional
        H⁺ density, in m⁻³. Default: ``electron_density`` (no H⁻ and no other ion).
    ion_temperature : float or array_like, optional
        Temperature of the H⁺ and the H⁻, in eV. Default: ``1``.
    atom_density : float or array_like, optional
        Density of H atoms, in m⁻³. Default: ``0``.
    atom_temperature : float or array_like, optional
        Temperature of the H atoms, in eV. Default: ``0.1``.

    Returns
    -------
    dict of float or numpy.ndarray
        ``{"electron detachment": ..., "mutual neutralization": ..., "atom detachment": ...}``,
        each ν in 1/s. The H⁻ lifetime is 1 over the sum.

    References
    ----------
    Janev, Reiter and Samm, Jül-4105, eqs. (58), (63) and (71).

    Examples
    --------
    >>> nu = destruction_frequencies(1e18, 1.0, atom_density=1e19)
    >>> print({k: f"{float(v):.3g}" for k, v in nu.items()})
    {'electron detachment': '1.47e+04', 'mutual neutralization': '6.09e+04', 'atom detachment': '2.42e+04'}
    """
    n_pos = electron_density if positive_ion_density is None else positive_ion_density
    rate_hh = maxwellian_rate_coefficient(
        lambda e: hydrogen_detachment_cross_section(e) + hydrogen_detachment_cross_section(e, "associative"),
        ion_temperature,
        mass_a=proton_mass,
        mass_b=proton_mass,
        temperature_b=atom_temperature,
        energy_range=(0.1, 2e4),
    )
    return {
        "electron detachment": _out(
            _array(electron_density) * electron_detachment_rate_coefficient(electron_temperature, ion_temperature)
        ),
        "mutual neutralization": _out(
            _array(n_pos) * mutual_neutralization_rate_coefficient(ion_temperature, ion_temperature)
        ),
        "atom detachment": _out(_array(atom_density) * rate_hh),
    }


# ---------------------------------------------------------------------------------------------
# Electronegative plasma
# ---------------------------------------------------------------------------------------------
def debye_length_electronegative(
    electron_density, electron_temperature, electronegativity, negative_ion_temperature, positive_ion_temperature=None
):
    """Compute the Debye length of an electron–H⁻–H⁺ plasma, 1/λ_D² = Σ_s 1/λ_s².

    Each Boltzmann species shields with its own length, λ_s² = ε₀ T_s / (n_s e²) (see
    :func:`~plasma_plots.theory.parameters.debye_length`). With n₋ = α n_e and n₊ = (1 + α) n_e,

    1/λ_D² = e² n_e / ε₀ · [1/T_e + α/T₋ + (1 + α)/T₊],

    from linearising Poisson's equation for Boltzmann electrons, negative and positive ions. For
    α → 0 and cold-ion shielding dropped (the default) λ_D is the electron Debye length.

    Parameters
    ----------
    electron_density : float or array_like
        n_e, in m⁻³.
    electron_temperature : float or array_like
        T_e, in eV.
    electronegativity : float or array_like
        α = n₋ / n_e.
    negative_ion_temperature : float or array_like
        T₋, in eV.
    positive_ion_temperature : float or array_like, optional
        T₊, in eV. Default: ``None``: the positive ions do not shield (hot ions), leaving the electrons
        and the H⁻. Pass T₊ to include them.

    Returns
    -------
    float or numpy.ndarray
        λ_D, in m. For T₊ = T₋ = T_e it is λ_e / √(2 + 2α).

    References
    ----------
    Derived from the Boltzmann relations of Kemaneci et al., arXiv:1612.07268, eq. (10), and
    Poisson's equation; sum of inverse squared species lengths as in Chen, Introduction to Plasma
    Physics, section 1.2 (electrons only).

    Examples
    --------
    1e18 m⁻³ electrons at 1 eV, α = 10 and H⁻ at 0.5 eV, without positive-ion shielding:

    >>> print(f"{debye_length_electronegative(1e18, 1.0, 10.0, 0.5) * 1e6:.4f} um")
    1.6222 um
    """
    n_e = _array(electron_density)
    alpha = _array(electronegativity)
    inverse_square = 1.0 / debye_length(n_e, electron_temperature) ** 2
    with np.errstate(divide="ignore"):  # alpha = 0: no H-, 1/inf = 0
        inverse_square = inverse_square + 1.0 / debye_length(alpha * n_e, negative_ion_temperature) ** 2
    if positive_ion_temperature is not None:
        inverse_square = inverse_square + 1.0 / debye_length((1.0 + alpha) * n_e, positive_ion_temperature) ** 2
    return _out(inverse_square**-0.5)


def sheath_edge_electronegativity(
    bulk_electronegativity, potential_drop, electron_temperature, negative_ion_temperature
):
    """Compute α_s = n₋/n_e at the sheath edge for Boltzmann electrons and negative ions.

    From T_e ∇ln n_e = T₋ ∇ln n₋ integrated through a presheath of potential drop V:
    n_e = n_e0 exp(−V/T_e) and n₋ = n₋0 exp(−V/T₋), so α_s = α₀ exp[−V (1/T₋ − 1/T_e)]. For
    T₋ < T_e the H⁻ are expelled faster than the electrons and α_s < α₀.

    Parameters
    ----------
    bulk_electronegativity : float or array_like
        α₀ = n₋/n_e in the bulk.
    potential_drop : float or array_like
        Potential drop V ≥ 0 from the bulk to the sheath edge, in V.
    electron_temperature : float or array_like
        T_e, in eV.
    negative_ion_temperature : float or array_like
        T₋, in eV.

    Returns
    -------
    float or numpy.ndarray
        α_s. It equals α₀ for T₋ = T_e or V = 0.

    References
    ----------
    Kemaneci et al., arXiv:1612.07268, eq. (10) (the Boltzmann condition) integrated in φ.

    Examples
    --------
    >>> print(f"{sheath_edge_electronegativity(5.0, 2.0, 2.0, 0.5):.4f}")
    0.2489
    """
    exponent = -_array(potential_drop) * (1.0 / _array(negative_ion_temperature) - 1.0 / _array(electron_temperature))
    return _out(_array(bulk_electronegativity) * np.exp(exponent))


def bohm_speed_electronegative(electron_temperature, electronegativity, negative_ion_temperature, mass_number=1):
    """Compute the Bohm speed of an electronegative plasma, u_B = √(T_e (1 + α_s) / (m_i (1 + γ α_s))).

    The Braithwaite–Allen criterion for cold positive ions and Boltzmann negative ions, with
    α_s = n₋/n_e at the sheath edge and γ = T_e/T₋: the electropositive
    :func:`~plasma_plots.theory.sheaths.bohm_speed` times √((1 + α_s)/(1 + γ α_s)). For γ > 1 the
    ions need less than the electropositive speed. For γ > 5 + √24 ≈ 9.9 the sheath-edge
    condition becomes multi-valued in the potential, and Braithwaite and Allen take the lower
    branch (the speed here is the criterion at a given α_s, not that selection).

    Parameters
    ----------
    electron_temperature : float or array_like
        T_e, in eV.
    electronegativity : float or array_like
        α_s = n₋/n_e at the sheath edge (see :func:`sheath_edge_electronegativity`).
    negative_ion_temperature : float or array_like
        T₋, in eV.
    mass_number : float or array_like, optional
        Positive-ion mass in proton masses. Default: ``1``.

    Returns
    -------
    float or numpy.ndarray
        u_B, in m/s. At α_s = 0 it is ``sheaths.bohm_speed(T_e)``.

    References
    ----------
    Kemaneci et al., arXiv:1612.07268, eq. (11); Braithwaite and Allen, J. Phys. D 21, 1733 (1988);
    the threshold γ = 5 + √24 for the multi-valued solution is from J. E. Allen, 31st EPS
    Conference on Plasma Physics (2004), P-1.036. Paul et al., arXiv:2002.05445, give the same
    factor as (n₋ + n_e) T₋ / (n₋ T_e + n_e T₋).

    Examples
    --------
    5 eV electrons, α_s = 1 and H⁻ at 1 eV (γ = 5):

    >>> print(f"{bohm_speed_electronegative(5.0, 1.0, 1.0):.4g} m/s")
    1.264e+04 m/s
    """
    alpha = _array(electronegativity)
    gamma = _array(electron_temperature) / _array(negative_ion_temperature)
    return _out(
        bohm_speed(electron_temperature, mass_number=mass_number) * np.sqrt((1.0 + alpha) / (1.0 + gamma * alpha))
    )


def ion_flux_electronegative(
    electron_density_edge, electronegativity, electron_temperature, negative_ion_temperature, mass_number=1
):
    """Compute the positive-ion flux Γ₊ = n₊ u_B = (1 + α_s) n_e u_B at the sheath edge.

    Parameters
    ----------
    electron_density_edge : float or array_like
        Electron density at the sheath edge, in m⁻³.
    electronegativity : float or array_like
        α_s = n₋/n_e at the sheath edge.
    electron_temperature : float or array_like
        T_e, in eV.
    negative_ion_temperature : float or array_like
        T₋, in eV.
    mass_number : float or array_like, optional
        Positive-ion mass in proton masses. Default: ``1``.

    Returns
    -------
    float or numpy.ndarray
        Γ₊, in m⁻² s⁻¹; multiply by e for the current density. The edge density is an input: the
        presheath that relates it to the bulk is not modelled.

    References
    ----------
    Kemaneci et al., arXiv:1612.07268, eq. (11) with the quasi-neutral n₊ = (1 + α_s) n_e.

    Examples
    --------
    >>> print(f"{ion_flux_electronegative(1e17, 1.0, 5.0, 1.0):.4g} m^-2 s^-1")
    2.527e+21 m^-2 s^-1
    """
    alpha = _array(electronegativity)
    speed = bohm_speed_electronegative(electron_temperature, alpha, negative_ion_temperature, mass_number)
    return _out((1.0 + alpha) * _array(electron_density_edge) * speed)


# ---------------------------------------------------------------------------------------------
# Beams and extraction
# ---------------------------------------------------------------------------------------------
def beam_survival_fraction(energy, h2_density=0.0, length=None, atom_density=0.0, positions=None, axis=-1):
    """Compute the fraction of an H⁻ beam that is not stripped by H₂ and H gas on its path.

    The product of two :func:`~plasma_plots.theory.beams.neutral_beam_attenuation`: the H₂ loss by
    :func:`h2_detachment_cross_section` at the laboratory energy E of the beam and the H loss by
    :func:`hydrogen_detachment_cross_section` (both channels) at the relative energy E/2.

    Parameters
    ----------
    energy : float or array_like
        Laboratory energy of the H⁻, in eV.
    h2_density : float or array_like, optional
        H₂ density, in m⁻³. Default: ``0``.
    length : float or array_like, optional
        Path length in m for constant densities; or give ``positions`` instead.
    atom_density : float or array_like, optional
        H density, in m⁻³. Default: ``0``.
    positions, axis
        See :func:`~plasma_plots.theory.beams.neutral_beam_attenuation`.

    Returns
    -------
    float or numpy.ndarray
        The surviving fraction of the beam. The H atom channel uses the 0.1 eV – 20 keV fit range,
        i.e. up to 40 keV beams.

    References
    ----------
    Janev, Reiter and Samm, Jül-4105, eqs. (159) and (71); the attenuation law of
    :func:`~plasma_plots.theory.beams.neutral_beam_attenuation`.

    Examples
    --------
    50 keV, 0.5 m of H₂ at 2.4e18 m⁻³ (about 0.01 Pa):

    >>> print(f"{beam_survival_fraction(50e3, h2_density=2.4e18, length=0.5):.4f}")
    0.9450
    """
    energy = _array(energy)
    sigma_h2 = h2_detachment_cross_section(energy)
    survival = neutral_beam_attenuation(h2_density, sigma_h2, length=length, positions=positions, axis=axis)
    sigma_h = hydrogen_detachment_cross_section(0.5 * energy) + hydrogen_detachment_cross_section(
        0.5 * energy, "associative"
    )
    return _out(
        survival * neutral_beam_attenuation(atom_density, sigma_h, length=length, positions=positions, axis=axis)
    )


def surface_production_current(flux, yield_, survival=1.0, area=1.0):
    """Compute the extractable H⁻ current I = e Y Γ S A from a surface-production flux.

    Parameters
    ----------
    flux : float or array_like
        Flux Γ of the converting particles (H⁺, H atoms, or the sum weighted by their yields) onto
        the surface, in m⁻² s⁻¹.
    yield_ : float or array_like
        H⁻ yield Y per incident particle.
    survival : float or array_like, optional
        Fraction S of the H⁻ that reaches the extraction aperture, e.g. from
        :func:`beam_survival_fraction`. Default: ``1``.
    area : float or array_like, optional
        Area A of the surface, in m². Default: ``1`` (a current density in A/m²).

    Returns
    -------
    float or numpy.ndarray
        I, in A.

    Examples
    --------
    A 1e22 m⁻² s⁻¹ atom flux with a 5 % yield and 70 % survival, on 1 cm²:

    >>> print(f"{surface_production_current(1e22, 0.05, 0.7, 1e-4) * 1e3:.4f} mA")
    5.6076 mA
    """
    return _out(elementary_charge * _array(yield_) * _array(flux) * _array(survival) * _array(area))
