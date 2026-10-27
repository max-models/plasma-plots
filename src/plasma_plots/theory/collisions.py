"""Coulomb collisions in SI units: the Coulomb logarithm, collision rates, Spitzer resistivity and
the slowing down of fast ions.

Densities are in m⁻³, temperatures and energies in eV, and ions have the mass ``mass_number × m_p``
and the charge ``charge_number × e`` (the conventions of :mod:`~plasma_plots.theory.parameters`).
Times are in s, frequencies in 1/s, resistivities in Ω m, energy loss rates in eV/s.

Conventions
-----------
* **Coulomb logarithm**: the NRL forms of :func:`coulomb_logarithm` unless you pass
  ``ln_lambda`` explicitly (a number or an array), so you can fix it as papers often do.
* **Fast-ion slowing down** (:func:`slowing_down_time`, :func:`critical_energy`,
  :func:`beam_stopping_fast_ion`): the classical result of Stix for an ion of speed v with
  v_Ti ≪ v ≪ v_Te, which loses energy to the electrons and the (cold) ions,

  .. math:: \\frac{dv}{dt} = -\\frac{v^3 + v_c^3}{\\tau_s v^2},

  with the Spitzer time τ_s on the electrons and the critical speed v_c, where the ion and
  electron drag are equal. No pitch-angle scattering, no energy diffusion, no orbits.

* **Pitch-angle evolution** (:func:`pitch_scattering_exponent`, :func:`pitch_angle_moments`):
  the fast ion also scatters in pitch ξ = v_∥/v on the plasma ions, with the Lorentz rate
  ν_d = Z_eff Γ/v³, Γ = n_e Z_b² e⁴ ln Λ/(4π ε₀² m_b²) (Z_eff = Σ n_j Z_j²/n_e), the form of the
  Monte-Carlo operators of arXiv:2111.11721, Eq. (D2)-(D7) and arXiv:2211.10900, Eq. (B5)-(B7).
  With the drag above, ν_d = β v_c³/(τ_s v³) with β = Z_eff/[Z], [Z] = A_b Σ_j (n_j/n_e) Z_j²/A_j,
  and the pitch moments follow in closed form (checked against RK4 in the tests).
* **Beam-driven current** (:func:`fast_ion_parallel_path`, :func:`electron_shielding_factor`,
  :func:`bootstrap_coefficient_l31`, :func:`beam_driven_current`): the circulating fast-ion
  current, shielded by the return current of the electrons, I = I_f [1 − (Z_b/Z_eff)(1 − G)],
  with G = L₃₁ of Sauter (arXiv:2211.10900, Eq. (C5)-(C6); Hu's note, Eq. (11)-(15)).

Not included
------------
Gaffey's own paper (1976) was not available to the author of this module, so the pitch
formula is derived from the operator above and checked numerically, not copied from it. The
large-aspect-ratio fit G = (1.55 + 0.85/Z) ε^½ − (0.20 + 1.55/Z) ε of Start and Cordey is
not implemented: its coefficients could not be confirmed from a source (use
:func:`bootstrap_coefficient_l31` or pass ``g``). Trapped fast ions, orbit widths and energy
diffusion are not modelled.

References
----------
J. D. Huba, NRL Plasma Formulary (Naval Research Laboratory, 2019), "Collision Rates" and
"Coulomb Logarithm".

L. Spitzer, Physics of Fully Ionized Gases, 2nd ed. (Interscience, 1962).

S. I. Braginskii, Transport processes in a plasma, Reviews of Plasma Physics 1, 205 (1965).

T. H. Stix, Heating of toroidal plasmas by neutral injection, Plasma Phys. 14, 367 (1972).

O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6, 2834 (1999), Eq. (14)-(17) (L₃₁), as
quoted in arXiv:2211.10900, Eq. (C5)-(C8), and Y. Hu, "Formulas for electron return current in
neutral beam current drive", https://youjunhu.github.io/research_notes/beam_driven_current.pdf,
Eq. (11)-(15).

Examples
--------
The Coulomb logarithm and Spitzer resistivity of a 1 keV, 1e19 m⁻³ hydrogen plasma:

>>> lnl = coulomb_logarithm(1e19, 1000.0)
>>> print(f"{lnl:.2f}, {spitzer_resistivity(1000.0, ln_lambda=lnl):.3e} ohm m")
15.94, 2.667e-08 ohm m
"""

from __future__ import annotations

import numpy as np

from plasma_plots.theory.parameters import (
    _ALPHA0,
    _ALPHA0_INVERSE_Z,
    _out,
    electron_mass,
    elementary_charge,
    ev_to_joule,
    proton_mass,
    vacuum_permittivity,
)

_e = elementary_charge
_eps0 = vacuum_permittivity


def _as_array(value):
    return np.asarray(value, dtype=float)


def coulomb_logarithm(density, temperature, kind="ei", charge_number=1):
    """Compute the Coulomb logarithm ln Λ of the NRL Plasma Formulary.

    Parameters
    ----------
    density : float or array_like
        Electron density n_e, in m⁻³.
    temperature : float or array_like
        Electron temperature T_e, in eV.
    kind : {"ei", "ee"}, optional
        ``"ei"``: electron-ion collisions, with the formulary's two regimes: for T_e below
        10 Z² eV, ln Λ = 23 − ln(n_e^½ Z T_e^(−3/2)); above, ln Λ = 24 − ln(n_e^½ T_e⁻¹)
        (n_e in cm⁻³). ``"ee"``: electron-electron collisions,
        ln Λ = 23.5 − ln(n_e^½ T_e^(−5/4)) − [10⁻⁵ + (ln T_e − 2)²/16]^½.
    charge_number : float or array_like, optional
        Ion charge number Z (only for ``"ei"``). Default: 1.

    Returns
    -------
    float or numpy.ndarray
        ln Λ.

    Notes
    -----
    The electron-ion forms assume T_i m_e/m_i < T_e. The two ``"ei"`` branches differ by 0.15
    where they meet (at T_e = 10 Z² eV), as in the formulary.

    References
    ----------
    NRL Plasma Formulary, "Coulomb Logarithm" (electron-ion, electron-electron).

    Examples
    --------
    >>> print(f"{coulomb_logarithm(1e19, 100.0):.2f}")  # 24 - ln(sqrt(1e13) / 100)
    13.64
    >>> # below 40 eV
    >>> print(f"{coulomb_logarithm(1e19, 5.0, charge_number=2):.2f}")
    9.75
    >>> print(f"{coulomb_logarithm(1e19, 100.0, kind='ee'):.2f}")
    13.64
    """
    n_cm3 = _as_array(density) * 1e-6
    temperature = _as_array(temperature)
    if kind == "ee":
        return _out(
            23.5
            - np.log(np.sqrt(n_cm3) * temperature**-1.25)
            - np.sqrt(1e-5 + (np.log(temperature) - 2.0) ** 2 / 16.0)
        )
    if kind != "ei":
        raise ValueError(f"kind must be 'ei' or 'ee', not {kind!r}")
    z = _as_array(charge_number)
    low = 23.0 - np.log(np.sqrt(n_cm3) * z * temperature**-1.5)
    high = 24.0 - np.log(np.sqrt(n_cm3) / temperature)
    return _out(np.where(temperature < 10.0 * z**2, low, high))


def _lnl(density, temperature, charge_number, ln_lambda):
    if ln_lambda is None:
        return _as_array(coulomb_logarithm(density, temperature, "ei", charge_number))
    return _as_array(ln_lambda)


def collision_frequencies(
    density,
    temperature,
    ion_temperature=None,
    mass_number=1.0,
    charge_number=1.0,
    ion_density=None,
    ln_lambda=None,
):
    """Compute the NRL electron and ion collision rates and the electron-ion momentum-transfer rate.

    Parameters
    ----------
    density : float or array_like
        Electron density n_e, in m⁻³.
    temperature : float or array_like
        Electron temperature T_e, in eV.
    ion_temperature : float or array_like, optional
        Ion temperature T_i, in eV. Default: T_e.
    mass_number : float or array_like, optional
        Ion mass in proton masses (μ). Default: 1.
    charge_number : float or array_like, optional
        Ion charge number Z. Default: 1.
    ion_density : float or array_like, optional
        Ion density n_i, in m⁻³. Default: n_e/Z.
    ln_lambda : float or array_like, optional
        Coulomb logarithm. Default: the ``"ei"`` form of :func:`coulomb_logarithm`.

    Returns
    -------
    dict of float or numpy.ndarray
        In 1/s: ``"electron"``, ν_e = √2 n_e e⁴ ln Λ / (12 π^(3/2) ε₀² m_e^½ T_e^(3/2)), the
        NRL electron collision rate (2.91e-6 n_e[cm⁻³] ln Λ T_e[eV]^(−3/2)); ``"ion"``,
        ν_i = n_i Z⁴ e⁴ ln Λ / (12 π^(3/2) ε₀² m_i^½ T_i^(3/2)), the NRL ion collision rate
        (4.80e-8 Z⁴ μ^(−½) n_i ln Λ T_i^(−3/2)); and ``"electron_ion"``, the momentum-transfer
        rate of the electrons on the ions, ν_ei = Z² n_i / n_e × ν_e, which is 1/τ_e of
        Braginskii.

    References
    ----------
    NRL Plasma Formulary, "Collision Rates"; Braginskii (1965), τ_e and τ_i.

    Examples
    --------
    >>> nu = collision_frequencies(
    ...     1e19, 100.0, ion_temperature=50.0, mass_number=2.0
    ... )
    >>> print(f"{nu['electron']:.3e} {nu['ion']:.3e} {nu['electron_ion']:.3e}")
    3.964e+05 1.308e+04 3.964e+05
    """
    n_e = _as_array(density)
    t_e = _as_array(temperature)
    t_i = t_e if ion_temperature is None else _as_array(ion_temperature)
    z = _as_array(charge_number)
    n_i = n_e / z if ion_density is None else _as_array(ion_density)
    lnl = _lnl(n_e, t_e, z, ln_lambda)
    m_i = _as_array(mass_number) * proton_mass
    common = _e**4 * lnl / (12 * np.pi**1.5 * _eps0**2)
    nu_e = (
        np.sqrt(2.0)
        * n_e
        * common
        / (np.sqrt(electron_mass) * (t_e * ev_to_joule) ** 1.5)
    )
    nu_i = n_i * z**4 * common / (np.sqrt(m_i) * (t_i * ev_to_joule) ** 1.5)
    return {
        "electron": _out(nu_e),
        "ion": _out(nu_i),
        "electron_ion": _out(nu_e * z**2 * n_i / n_e),
    }


def spitzer_resistivity(temperature, z_eff=1.0, ln_lambda=None, density=None):
    """Compute the Spitzer parallel resistivity η_∥ = α₀(Z) Z_eff m_e ν_e / (n_e e²).

    Here ν_e is the electron collision rate of :func:`collision_frequencies` for Z = 1, so
    η_∥ = α₀(Z_eff) × 1.03e-4 Z_eff ln Λ T_e[eV]^(−3/2) Ω m, which is 1.67e-9 ln Λ Z_eff /
    T_e[keV]^(3/2) Ω m for Z_eff = 1 (1.65e-9 with the NRL's 0.51 in place of 0.5129).

    Parameters
    ----------
    temperature : float or array_like
        Electron temperature T_e, in eV.
    z_eff : float or array_like, optional
        Effective charge Σ n_j Z_j² / n_e. Default: 1.
    ln_lambda : float or array_like, optional
        Coulomb logarithm. Default: computed from ``density`` (``"ei"`` form, with Z = ``z_eff``).
    density : float or array_like, optional
        Electron density, in m⁻³. Needed only without ``ln_lambda``.

    Returns
    -------
    float or numpy.ndarray
        η_∥ in Ω m (it does not depend on the density except through ln Λ).

    Notes
    -----
    α₀ is the Braginskii/Spitzer coefficient of the parallel conductivity, 0.5129, 0.4408,
    0.3965, 0.3752 for Z = 1, 2, 3, 4 and 0.2949 for Z → ∞, interpolated linearly in 1/Z
    for non-integer Z_eff (an approximation between the tabulated charges). The NRL Plasma Formulary quotes
    η_∥ = 0.51 η_⊥ = 5.2e-5 Z ln Λ T_e^(−3/2) Ω m (Z = 1), 1.5 % below this value (rounding and 0.51 against 0.5129).

    References
    ----------
    Spitzer (1962); Braginskii (1965); NRL Plasma Formulary, "Transport Coefficients".

    Examples
    --------
    >>> # about 1.65e-9 * 17
    >>> print(f"{spitzer_resistivity(1000.0, ln_lambda=17.0):.3e} ohm m")
    2.844e-08 ohm m
    >>> print(
    ...     f"{spitzer_resistivity(1000.0, z_eff=2.0, ln_lambda=17.0):.3e} ohm m"
    ... )
    4.888e-08 ohm m
    """
    t_e = _as_array(temperature)
    z = np.maximum(_as_array(z_eff), 1.0)
    if ln_lambda is None:
        if density is None:
            raise ValueError(
                "give ln_lambda or the electron density to compute it from"
            )
        ln_lambda = coulomb_logarithm(density, t_e, "ei", z)
    alpha0 = np.interp(1.0 / z, _ALPHA0_INVERSE_Z, _ALPHA0)
    # m_e / (n_e e² τ_e) with τ_e of Braginskii, n_i Z² = n_e Z_eff:
    prefactor = (
        np.sqrt(2.0) * np.sqrt(electron_mass) * _e**2 / (12 * np.pi**1.5 * _eps0**2)
    )
    eta = alpha0 * z * prefactor * _as_array(ln_lambda) / (t_e * ev_to_joule) ** 1.5
    return _out(eta)


def _ion_drag_sum(ions, mass_number, charge_number):
    """Σ_j (n_j/n_e) Z_j² / A_j over the plasma ions."""
    if ions is None:
        z = _as_array(charge_number)
        return (1.0 / z) * z**2 / _as_array(mass_number)
    return sum(f * z**2 / a for f, z, a in ions)


def slowing_down_time(
    density,
    temperature,
    beam_mass_number=1.0,
    beam_charge_number=1.0,
    ln_lambda=None,
    energy=None,
    final_energy=0.0,
    critical_energy_ev=None,
):
    """Compute the Spitzer slowing-down time τ_s of a fast ion on the electrons, or the time to slow down.

    τ_s = 3 (2π)^(3/2) ε₀² m_b T_e^(3/2) / (n_e Z_b² e⁴ m_e^½ ln Λ), which is
    6.3e8 A_b T_e[eV]^(3/2) / (Z_b² n_e[cm⁻³] ln Λ) s: the time in which the ion's speed would
    fall by a factor e if only the electrons acted. With ``energy`` given, return the time to
    slow from that energy to ``final_energy`` in the presence of the ion drag as well (Stix):
    t = (τ_s/3) ln[(1 + (E/E_c)^(3/2)) / (1 + (E_f/E_c)^(3/2))], from dv/dt = −(v³ + v_c³)/(τ_s v²).

    Parameters
    ----------
    density : float or array_like
        Electron density n_e, in m⁻³.
    temperature : float or array_like
        Electron temperature T_e, in eV.
    beam_mass_number, beam_charge_number : float or array_like, optional
        Mass (proton masses) and charge number of the fast ion. Default: 1.
    ln_lambda : float or array_like, optional
        Coulomb logarithm. Default: the ``"ei"`` form of :func:`coulomb_logarithm`.
    energy : float or array_like, optional
        Initial energy E of the fast ion, in eV. If given, return the time to reach
        ``final_energy`` instead of τ_s. Needs ``critical_energy_ev``.
    final_energy : float or array_like, optional
        Final energy, in eV. Default: 0 (the time to thermalise, in this model).
    critical_energy_ev : float or array_like, optional
        The critical energy E_c, in eV (from :func:`critical_energy`). Use ``np.inf`` for
        the electron drag alone.

    Returns
    -------
    float or numpy.ndarray
        τ_s in s, or the slowing-down time in s.

    References
    ----------
    Spitzer (1962), p. 134; Stix (1972), Eq. (13) and (16); NRL Plasma Formulary.

    Examples
    --------
    >>> tau = slowing_down_time(1e19, 1000.0, beam_mass_number=2.0, ln_lambda=17.0)
    >>> print(f"{tau * 1e3:.2f} ms")
    235.04 ms
    >>> ec = critical_energy(1000.0, beam_mass_number=2.0, ion_mass_number=2.0)
    >>> t = slowing_down_time(
    ...     1e19, 1000.0, 2.0, ln_lambda=17.0, energy=80e3, critical_energy_ev=ec
    ... )
    >>> print(f"{ec / 1e3:.1f} keV, {t * 1e3:.1f} ms")
    18.7 keV, 179.5 ms
    """
    n_e = _as_array(density)
    t_e = _as_array(temperature)
    z_b = _as_array(beam_charge_number)
    lnl = _lnl(n_e, t_e, 1.0, ln_lambda)
    tau_s = (
        3.0
        * (2.0 * np.pi) ** 1.5
        * _eps0**2
        * _as_array(beam_mass_number)
        * proton_mass
        * (t_e * ev_to_joule) ** 1.5
        / (n_e * z_b**2 * _e**4 * np.sqrt(electron_mass) * lnl)
    )
    if energy is None:
        return _out(tau_s)
    if critical_energy_ev is None:
        raise ValueError("critical_energy_ev is needed to slow down from an energy")
    e_c = _as_array(critical_energy_ev)
    ratio = (_as_array(energy) / e_c) ** 1.5
    ratio_f = (_as_array(final_energy) / e_c) ** 1.5
    return _out(tau_s / 3.0 * np.log((1.0 + ratio) / (1.0 + ratio_f)))


def critical_energy(
    temperature,
    beam_mass_number=1.0,
    ion_mass_number=1.0,
    ion_charge_number=1.0,
    ions=None,
):
    """Compute the critical energy E_c, where a fast ion's drag on the ions equals that on the electrons.

    E_c = ½ m_b v_c² with v_c³ = (3√π/4) m_e v_Te³ Σ_j n_j Z_j² / (n_e m_j) and
    v_Te = √(2 T_e/m_e), i.e. E_c = 14.8 T_e A_b [Σ_j (n_j/n_e) Z_j² / A_j]^(2/3). It does not
    depend on the density. Above E_c the fast ion heats the electrons, below it the ions.

    Parameters
    ----------
    temperature : float or array_like
        Electron temperature T_e, in eV.
    beam_mass_number : float or array_like, optional
        Mass of the fast ion A_b, in proton masses. Default: 1.
    ion_mass_number, ion_charge_number : float or array_like, optional
        Mass number and charge number of a single plasma ion species, with n_i = n_e/Z.
        Defaults: 1 (hydrogen).
    ions : sequence of (float, float, float), optional
        Several plasma ion species as (n_j/n_e, Z_j, A_j) tuples, in place of the single
        species (e.g. ``[(0.45, 1, 2), (0.45, 1, 3), (0.025, 4, 4)]``).

    Returns
    -------
    float or numpy.ndarray
        E_c in eV.

    References
    ----------
    Stix (1972); the 14.8 is Wesson, Tokamaks, and the NRL value of the same integral.

    Examples
    --------
    >>> print(
    ...     f"{critical_energy(1000.0, beam_mass_number=2.0, ion_mass_number=2.0) / 1e3:.2f} keV"
    ... )
    18.65 keV
    >>> # a D beam in a 50:50 D-T plasma (the sum of 0.5/2 + 0.5/3 is 5/12)
    >>> print(
    ...     f"{critical_energy(10e3, 2.0, ions=[(0.5, 1, 2), (0.5, 1, 3)]) / 1e3:.0f} keV"
    ... )
    165 keV
    """
    s = _ion_drag_sum(ions, ion_mass_number, ion_charge_number)
    factor = 3.0 * np.sqrt(np.pi) / 4.0 * electron_mass / proton_mass
    # v_c^3 = (3 sqrt(pi)/4) (m_e/m_p) v_Te^3 s, with s = sum_j (n_j/n_e) Z_j^2 / A_j
    t_j = _as_array(temperature) * ev_to_joule
    v_c3 = factor * (2.0 * t_j / electron_mass) ** 1.5 * s
    e_c = 0.5 * _as_array(beam_mass_number) * proton_mass * v_c3 ** (2.0 / 3.0)
    return _out(e_c / ev_to_joule)


def beam_stopping_fast_ion(
    energy,
    density,
    temperature,
    beam_mass_number=1.0,
    beam_charge_number=1.0,
    ion_mass_number=1.0,
    ion_charge_number=1.0,
    ions=None,
    ln_lambda=None,
    time=None,
):
    """Compute the energy loss of a fast ion on electrons and ions (Stix), or its energy after a time.

    With the ion drag and electron drag of the model of the module docstring,
    dE/dt = −(2E/τ_s) [1 + (E_c/E)^(3/2)]: the first term is the loss to the electrons, the
    second to the ions. Given ``time`` t, the energy after t follows from integrating dv/dt:
    v³ = (v₀³ + v_c³) exp(−3t/τ_s) − v_c³, so E = E₀ [(1 + (E_c/E₀)^(3/2)) e^(−3t/τ_s) −
    (E_c/E₀)^(3/2)]^(2/3), and 0 once that is negative (the ion has thermalised).

    Parameters
    ----------
    energy : float or array_like
        Energy E of the fast ion, in eV.
    density, temperature : float or array_like
        Electron density (m⁻³) and temperature (eV).
    beam_mass_number, beam_charge_number : float or array_like, optional
        Mass (proton masses) and charge number of the fast ion. Default: 1.
    ion_mass_number, ion_charge_number, ions : optional
        The plasma ions, as in :func:`critical_energy`.
    ln_lambda : float or array_like, optional
        Coulomb logarithm. Default: the ``"ei"`` form (Z = 1).
    time : float or array_like, optional
        If given, return the energy after this time, in s.

    Returns
    -------
    dict or float or numpy.ndarray
        Without ``time``: the energy loss rates in eV/s (positive numbers) as a dict with
        ``"electrons"``, ``"ions"`` and ``"total"``. With ``time``: the energy in eV.

    References
    ----------
    Stix (1972).

    Examples
    --------
    >>> kw = dict(
    ...     density=1e19,
    ...     temperature=1000.0,
    ...     beam_mass_number=2.0,
    ...     ion_mass_number=2.0,
    ...     ln_lambda=17.0,
    ... )
    >>> loss = beam_stopping_fast_ion(80e3, **kw)
    >>> print(f"{loss['electrons']:.3e} {loss['ions']:.3e} eV/s")
    6.807e+05 7.664e+04 eV/s
    >>> print(
    ...     f"{beam_stopping_fast_ion(80e3, time=0.02, **kw) / 1e3:.1f} keV after 20 ms"
    ... )
    66.0 keV after 20 ms
    """
    tau_s = _as_array(
        slowing_down_time(
            density, temperature, beam_mass_number, beam_charge_number, ln_lambda
        )
    )
    e_c = _as_array(
        critical_energy(
            temperature, beam_mass_number, ion_mass_number, ion_charge_number, ions
        )
    )
    e0 = _as_array(energy)
    if time is None:
        electrons = 2.0 * e0 / tau_s
        ions_loss = electrons * (e_c / e0) ** 1.5
        return {
            "electrons": _out(electrons),
            "ions": _out(ions_loss),
            "total": _out(electrons + ions_loss),
        }
    ratio = (e_c / e0) ** 1.5
    cube = (1.0 + ratio) * np.exp(-3.0 * _as_array(time) / tau_s) - ratio
    return _out(e0 * np.maximum(cube, 0.0) ** (2.0 / 3.0))


def _z_effective(ions, charge_number):
    if ions is None:
        return _as_array(charge_number)
    return sum(f * z**2 for f, z, _ in ions)


def pitch_scattering_exponent(
    beam_mass_number=1.0, ion_mass_number=1.0, ion_charge_number=1.0, ions=None
):
    """Compute the exponent β = Z_eff/[Z] of the pitch-angle evolution of a slowing-down fast ion.

    The pitch scattering rate on the plasma ions is ν_d = Z_eff Γ/v³ and the energy drag on them is
    Γ [Z]/v² per unit mass, with Z_eff = Σ_j (n_j/n_e) Z_j² and [Z] = A_b Σ_j (n_j/n_e) Z_j²/A_j
    (the sum in :func:`critical_energy`). Hence ν_d = β v_c³/(τ_s v³). For one plasma ion species
    with n_i = n_e/Z, β = A_i/A_b: 1 for a beam in a plasma of its own isotope, independent of Z.
    Impurities raise Z_eff faster than [Z]: β grows.

    Parameters
    ----------
    beam_mass_number : float or array_like, optional
        Mass of the fast ion A_b, in proton masses. Default: 1.
    ion_mass_number, ion_charge_number : float or array_like, optional
        A single plasma ion species, as in :func:`critical_energy`. Defaults: 1.
    ions : sequence of (float, float, float), optional
        Several species as (n_j/n_e, Z_j, A_j) tuples, in place of the single species.

    Returns
    -------
    float or numpy.ndarray
        β (dimensionless).

    References
    ----------
    arXiv:2111.11721, Eq. (D3)-(D7); arXiv:2211.10900, Eq. (B5)-(B7).

    Examples
    --------
    >>> print(f"{pitch_scattering_exponent(2.0, 2.0):.2f}")  # D in D
    1.00
    >>> # D beam in a D plasma with 5 % carbon 6+ (n_C/n_e = 0.05)
    >>> ions = [(0.7, 1, 2), (0.05, 6, 12)]
    >>> print(f"{pitch_scattering_exponent(2.0, ions=ions):.2f}")
    2.50
    """
    s = _ion_drag_sum(ions, ion_mass_number, ion_charge_number)
    z_eff = _z_effective(ions, ion_charge_number)
    return _out(z_eff / (_as_array(beam_mass_number) * s))


def _pitch_ratio(energy, initial_energy, critical_energy_ev):
    """R = [v³ (v₀³ + v_c³)] / [v₀³ (v³ + v_c³)] from the energies."""
    e_c = _as_array(critical_energy_ev)
    u3 = (_as_array(energy) / e_c) ** 1.5
    u03 = (_as_array(initial_energy) / e_c) ** 1.5
    return u3 * (1.0 + u03) / (u03 * (1.0 + u3))


def pitch_angle_moments(
    energy, initial_energy, critical_energy_ev, exponent=1.0, initial_pitch=1.0
):
    """Compute the pitch ξ = v_∥/v of a fast ion that slows down from E₀ to E: its mean, mean square and width.

    Pitch scattering on the ions with the Lorentz operator (ν_d/2) ∂_ξ (1 − ξ²) ∂_ξ gives
    d⟨ξ⟩/dt = −ν_d ⟨ξ⟩ and d⟨ξ²⟩/dt = ν_d (1 − 3⟨ξ²⟩) exactly, and the speed follows the drag of
    :func:`beam_stopping_fast_ion`, dv/dt = −(v³ + v_c³)/(τ_s v²). With ν_d = β v_c³/(τ_s v³) and
    R = v³ (v₀³ + v_c³) / [v₀³ (v³ + v_c³)] = (E/E₀)^(3/2) (1 + (E₀/E_c)^(3/2)) / (1 + (E/E_c)^(3/2)):

    .. math:: \\langle\\xi\\rangle = \\xi_0 R^{\\beta/3}, \\qquad
              \\langle\\xi^2\\rangle = \\tfrac13 + (\\xi_0^2 - \\tfrac13) R^{\\beta}

    for ions that start with the single pitch ξ₀. The width is σ = [⟨ξ²⟩ − ⟨ξ⟩²]^½, which
    grows from 0 to 1/√3 (isotropy) as the ion slows. Above E_c, R ≈ 1: the ion keeps its pitch
    while it loses most of its energy to the electrons; below E_c it scatters (R ∝ E^(3/2)).

    Parameters
    ----------
    energy : float or array_like
        Energy E of the ion, in eV (E ≤ E₀).
    initial_energy : float or array_like
        Birth energy E₀, in eV.
    critical_energy_ev : float or array_like
        The critical energy E_c, in eV (:func:`critical_energy`).
    exponent : float or array_like, optional
        β of :func:`pitch_scattering_exponent`. Default: 1.
    initial_pitch : float or array_like, optional
        Birth pitch ξ₀ of every ion. Default: 1 (parallel).

    Returns
    -------
    dict of float or numpy.ndarray
        ``"mean"`` ⟨ξ⟩, ``"mean_square"`` ⟨ξ²⟩ and ``"width"`` σ.

    Notes
    -----
    The β exponent follows from the Monte-Carlo operator of arXiv:2111.11721 and
    arXiv:2211.10900 (see :func:`pitch_scattering_exponent`) and is checked against an RK4
    integration of the moment equations in the tests. Gaffey (1976) was not read.

    References
    ----------
    arXiv:2111.11721, Eq. (D2)-(D7); arXiv:2211.10900, Eq. (B5)-(B7); Stix (1972) for the drag.

    Examples
    --------
    >>> ec = critical_energy(1000.0, 2.0, 2.0)
    >>> m = pitch_angle_moments(ec, 80e3, ec, exponent=1.0)
    >>> print(f"{m['mean']:.3f} {m['width']:.3f}")  # at E_c
    0.822 0.167
    >>> m = pitch_angle_moments(1000.0, 80e3, ec, exponent=1.0)
    >>> print(f"{m['mean']:.3f} {m['width']:.3f}")  # nearly isotropic
    0.239 0.534
    """
    r = _pitch_ratio(energy, initial_energy, critical_energy_ev)
    xi0 = _as_array(initial_pitch)
    beta = _as_array(exponent)
    mean = xi0 * r ** (beta / 3.0)
    mean_square = 1.0 / 3.0 + (xi0**2 - 1.0 / 3.0) * r**beta
    width = np.sqrt(np.maximum(mean_square - mean**2, 0.0))
    return {"mean": _out(mean), "mean_square": _out(mean_square), "width": _out(width)}


_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(200)


def fast_ion_parallel_path(
    energy,
    density,
    temperature,
    beam_mass_number=1.0,
    beam_charge_number=1.0,
    ion_mass_number=1.0,
    ion_charge_number=1.0,
    ions=None,
    ln_lambda=None,
    initial_pitch=1.0,
):
    """Compute ∫ v_∥ dt, the distance a fast ion travels along B while it slows down from E to rest.

    With dt = τ_s v² dv/(v³ + v_c³) and ⟨ξ⟩(v) of :func:`pitch_angle_moments`,
    L = ξ₀ τ_s v_c ∫₀^x u³/(1 + u³) [u³ (1 + x³)/(x³ (1 + u³))]^(β/3) du, with x = v₀/v_c.
    The integral is done by 200-point Gauss-Legendre quadrature.

    Parameters
    ----------
    energy : float or array_like
        Birth energy E₀, in eV.
    density, temperature : float or array_like
        Electron density (m⁻³) and temperature (eV).
    beam_mass_number, beam_charge_number : float or array_like, optional
        Mass (proton masses) and charge number of the fast ion. Default: 1.
    ion_mass_number, ion_charge_number, ions : optional
        The plasma ions, as in :func:`critical_energy`.
    ln_lambda : float or array_like, optional
        Coulomb logarithm. Default: the ``"ei"`` form (Z = 1).
    initial_pitch : float or array_like, optional
        Birth pitch ξ₀. Default: 1.

    Returns
    -------
    float or numpy.ndarray
        L in m.

    Examples
    --------
    >>> kw = dict(
    ...     density=1e19,
    ...     temperature=1000.0,
    ...     beam_mass_number=2.0,
    ...     ion_mass_number=2.0,
    ...     ln_lambda=17.0,
    ... )
    >>> print(f"{fast_ion_parallel_path(80e3, **kw) / 1e3:.1f} km")
    276.7 km
    """
    tau_s = _as_array(
        slowing_down_time(
            density, temperature, beam_mass_number, beam_charge_number, ln_lambda
        )
    )
    e_c = _as_array(
        critical_energy(
            temperature, beam_mass_number, ion_mass_number, ion_charge_number, ions
        )
    )
    beta = _as_array(
        pitch_scattering_exponent(
            beam_mass_number, ion_mass_number, ion_charge_number, ions
        )
    )
    v_c = np.sqrt(2.0 * e_c * ev_to_joule / (_as_array(beam_mass_number) * proton_mass))
    x = np.sqrt(_as_array(energy) / e_c)
    xe, be = x[..., None], beta[..., None]
    u = 0.5 * xe * (_GL_NODES + 1.0)
    pitch = (u**3 * (1.0 + xe**3) / (xe**3 * (1.0 + u**3))) ** (be / 3.0)
    integral = 0.5 * x * np.sum(_GL_WEIGHTS * u**3 / (1.0 + u**3) * pitch, axis=-1)
    return _out(_as_array(initial_pitch) * tau_s * v_c * integral)


def bootstrap_coefficient_l31(z_eff, trapped_fraction, collisionality=0.0):
    """Compute the coefficient L₃₁ of Sauter et al., which is the trapped-electron correction G of the beam current.

    L₃₁ = [1 + 1.4/(Z_eff + 1)] X − 1.9/(Z_eff + 1) X² + 0.3/(Z_eff + 1) X³ + 0.2/(Z_eff + 1) X⁴,
    X = f_t / [1 + (1 − 0.1 f_t) √ν_e* + 0.5 (1 − f_t) ν_e*/Z_eff].

    Parameters
    ----------
    z_eff : float or array_like
        Effective charge of the plasma ions.
    trapped_fraction : float or array_like
        Effective trapped fraction f_t of the flux surface (0 for a straight plasma).
    collisionality : float or array_like, optional
        Electron collisionality ν_e* (0: banana regime, the default). The definition of ν_e*
        differs between the two sources read (Z_eff included or not), so it is an input.

    Returns
    -------
    float or numpy.ndarray
        L₃₁ = G, between 0 (no trapped electrons) and about 1.

    References
    ----------
    arXiv:2211.10900, Eq. (C6); Hu, note on the electron return current, Eq. (11)-(12) (the form
    of Sauter et al., Phys. Plasmas 6, 2834 (1999); Hu notes a sign error in the original's use).

    Examples
    --------
    >>> print(f"{bootstrap_coefficient_l31(1.0, 0.3):.3f}")
    0.429
    >>> print(f"{bootstrap_coefficient_l31(2.0, 0.3, collisionality=1.0):.3f}")
    0.193
    """
    z = _as_array(z_eff)
    f_t = _as_array(trapped_fraction)
    nu = _as_array(collisionality)
    x = f_t / (1.0 + (1.0 - 0.1 * f_t) * np.sqrt(nu) + 0.5 * (1.0 - f_t) * nu / z)
    d = z + 1.0
    return _out((1.0 + 1.4 / d) * x - 1.9 / d * x**2 + 0.3 / d * x**3 + 0.2 / d * x**4)


def electron_shielding_factor(beam_charge_number, z_eff, g=0.0):
    """Compute F = j/j_f = 1 − (Z_b/Z_eff)(1 − G), the ratio of the net beam-driven current to the fast-ion current.

    The electrons are dragged along by the fast ions and carry a return current −(Z_b/Z_eff)(1 − G)
    times the fast-ion current (for a beam speed well below v_Te). G is the trapped-electron
    correction: 0 in a straight (uniform) plasma, where F = 1 − Z_b/Z_eff and the current of a
    beam in a plasma of its own charge (Z_eff = Z_b) cancels completely.

    Parameters
    ----------
    beam_charge_number, z_eff : float or array_like
        Charge number of the fast ion and effective charge of the plasma ions.
    g : float or array_like, optional
        Trapped-electron correction G (e.g. :func:`bootstrap_coefficient_l31`). Default: 0.

    Returns
    -------
    float or numpy.ndarray
        F.

    References
    ----------
    arXiv:2211.10900, Eq. (C5); Hu (note above), Eq. (14)-(16); Ohkawa, Nucl. Fusion 10, 185 (1970);
    Start and Cordey, Phys. Fluids 23, 1477 (1980).

    Examples
    --------
    >>> print(f"{electron_shielding_factor(1.0, 2.0):.3f}")
    0.500
    >>> print(f"{electron_shielding_factor(1.0, 1.0, g=0.3):.3f}")
    0.300
    """
    return _out(
        1.0 - _as_array(beam_charge_number) / _as_array(z_eff) * (1.0 - _as_array(g))
    )


def beam_driven_current(
    birth_rate,
    major_radius,
    energy,
    density,
    temperature,
    beam_mass_number=1.0,
    beam_charge_number=1.0,
    ion_mass_number=1.0,
    ion_charge_number=1.0,
    ions=None,
    ln_lambda=None,
    initial_pitch=1.0,
    g=0.0,
):
    """Compute the beam-driven current I_b = I_f [1 − (Z_b/Z_eff)(1 − G)] of fast ions born at one energy and pitch.

    Each ion born with parallel speed circulates and carries Z_b e v_∥/(2π R) of current, so a
    birth rate S gives the fast-ion current I_f = Z_b e S L/(2π R), L = ∫ v_∥ dt of
    :func:`fast_ion_parallel_path` (the Ohkawa estimate with the slowing-down and the pitch
    evolution). The electron return current reduces it by :func:`electron_shielding_factor`.
    A local, large-aspect-ratio estimate: no orbits, trapped fast ions or losses.

    Parameters
    ----------
    birth_rate : float or array_like
        Number of fast ions born per second, S.
    major_radius : float or array_like
        Major radius R of the surface, in m.
    energy, density, temperature, beam_mass_number, beam_charge_number, ion_mass_number, ion_charge_number, ions, ln_lambda, initial_pitch :
        As in :func:`fast_ion_parallel_path`.
    g : float or array_like, optional
        Trapped-electron correction G. Default: 0 (uniform plasma).

    Returns
    -------
    dict of float or numpy.ndarray
        ``"fast_ion"`` I_f in A, ``"shielding"`` F, ``"net"`` I_b = F I_f in A.

    Examples
    --------
    >>> kw = dict(
    ...     density=1e19,
    ...     temperature=1000.0,
    ...     beam_mass_number=2.0,
    ...     ion_mass_number=2.0,
    ...     ion_charge_number=2.0,
    ...     ln_lambda=17.0,
    ... )
    >>> # 1 MW of 80 keV deuterons at R = 1.7 m, in a plasma of Z = 2 ions (Z_eff = 2)
    >>> s = 1e6 / (80e3 * ev_to_joule)
    >>> out = beam_driven_current(s, 1.7, 80e3, **kw)
    >>> print(
    ...     f"{out['fast_ion'] / 1e3:.1f} kA {out['shielding']:.2f} {out['net'] / 1e3:.1f} kA"
    ... )
    247.4 kA 0.50 123.7 kA
    """
    path = fast_ion_parallel_path(
        energy,
        density,
        temperature,
        beam_mass_number,
        beam_charge_number,
        ion_mass_number,
        ion_charge_number,
        ions,
        ln_lambda,
        initial_pitch,
    )
    z_eff = _z_effective(ions, ion_charge_number)
    i_f = (
        _as_array(beam_charge_number)
        * _e
        * _as_array(birth_rate)
        * path
        / (2.0 * np.pi * _as_array(major_radius))
    )
    shield = _as_array(electron_shielding_factor(beam_charge_number, z_eff, g))
    return {"fast_ion": _out(i_f), "shielding": _out(shield), "net": _out(i_f * shield)}
