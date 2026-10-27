"""Plasma parameters in SI units: frequencies, lengths, speeds and Struphy's units.

Densities are in m⁻³, magnetic fields in T, temperatures in eV (k_B T / e, so 1 eV is
11604.5 K) unless a function says otherwise, and the results are in SI units: angular
frequencies in rad/s, lengths in m, speeds in m/s. Ions have the mass ``mass_number × m_p``
(proton masses, as in Struphy and the NRL Plasma Formulary) and the charge
``charge_number × e``.

Constants
---------
CODATA 2018 values, the same as Struphy's ``struphy.physics.physics.ConstantsOfNature``:

* ``elementary_charge`` (C), ``electron_mass`` and ``proton_mass`` (kg),
* ``vacuum_permittivity`` ε₀ (F/m), ``vacuum_permeability`` μ₀ (N/A²),
  ``speed_of_light`` c (m/s), ``boltzmann_constant`` k_B (J/K),
* ``ev_to_joule`` (J per eV) and ``ev_to_kelvin`` (K per eV).

Conventions
-----------
* **Thermal speed**: v_th = √(T/m) by default (the standard deviation of a 1-D Maxwellian, the
  NRL and Struphy convention); :func:`thermal_speed` also gives √(2T/m) and the mean speed.
* **Cyclotron frequency**: Ω = q|B|/m, which has the sign of the charge you pass. The default
  charge is +e, so the default is the magnitude; pass ``charge=-elementary_charge`` for the
  signed electron frequency.

References
----------
J. D. Huba, NRL Plasma Formulary (Naval Research Laboratory, 2019).

F. F. Chen, Introduction to Plasma Physics and Controlled Fusion, 3rd ed. (Springer, 2016).

Examples
--------
The electron plasma frequency and Debye length of a 1e19 m⁻³, 100 eV plasma:

>>> print(
...     f"{plasma_frequency(1e19):.3e} rad/s, {debye_length(1e19, 100.0):.3e} m"
... )
1.784e+11 rad/s, 2.351e-05 m
"""

from __future__ import annotations

import numpy as np

elementary_charge = 1.602176634e-19
electron_mass = 9.1093837015e-31
proton_mass = 1.67262192369e-27
vacuum_permittivity = 8.8541878128e-12
vacuum_permeability = 1.25663706212e-6
speed_of_light = 299792458.0
boltzmann_constant = 1.380649e-23
ev_to_joule = elementary_charge
ev_to_kelvin = elementary_charge / boltzmann_constant

_e = elementary_charge
_THERMAL_CONVENTIONS = {"sqrt(T/m)": 1.0, "sqrt(2T/m)": 2.0, "mean": 8 / np.pi}

#: The SI units of the entries of :func:`struphy_units`.
STRUPHY_UNIT_SYMBOLS = {
    "x": "m",
    "B": "T",
    "n": "m⁻³",
    "kBT": "keV",
    "v": "m/s",
    "t": "s",
    "frequency": "1/s",
    "p": "Pa",
    "rho": "kg/m³",
    "j": "A/m²",
}


def _out(value):
    return np.asarray(value, dtype=float)[()]


def plasma_frequency(density, mass=electron_mass, charge=elementary_charge):
    """Compute the plasma frequency ω_p = √(n q² / (ε₀ m)).

    Parameters
    ----------
    density : float or array_like
        Number density n of the species, in m⁻³.
    mass : float or array_like, optional
        Particle mass m, in kg. Default: the electron mass.
    charge : float or array_like, optional
        Particle charge q, in C (its sign doesn't matter). Default: e.

    Returns
    -------
    float or numpy.ndarray
        ω_p in rad/s (divide by 2π for Hz).

    References
    ----------
    NRL Plasma Formulary: f_pe = ω_pe/2π = 8.98 kHz × √(n_e / cm⁻³).

    Examples
    --------
    >>> print(f"{plasma_frequency(1e18) / (2 * np.pi):.4g} Hz")
    8.979e+09 Hz
    >>> # deuterons
    >>> print(f"{plasma_frequency(1e20, mass=2 * proton_mass):.4g} rad/s")
    9.309e+09 rad/s
    """
    density = np.asarray(density, dtype=float)
    return _out(np.sqrt(density * np.square(charge) / (vacuum_permittivity * mass)))


def cyclotron_frequency(field, mass=electron_mass, charge=elementary_charge):
    """Compute the cyclotron (gyro-) frequency Ω = q|B|/m.

    Parameters
    ----------
    field : float or array_like
        Magnetic field B, in T (its sign doesn't matter).
    mass : float or array_like, optional
        Particle mass m, in kg. Default: the electron mass.
    charge : float or array_like, optional
        Particle charge q, in C. Ω has its sign: the default +e gives the magnitude, −e the
        signed electron frequency (negative, for the left-handed electron gyration).

    Returns
    -------
    float or numpy.ndarray
        Ω in rad/s.

    References
    ----------
    NRL Plasma Formulary: f_ce = Ω_e/2π = 2.80 MHz × B/G = 28.0 GHz × B/T.

    Examples
    --------
    >>> print(f"{cyclotron_frequency(1.0) / (2 * np.pi):.4g} Hz")
    2.799e+10 Hz
    >>> print(f"{cyclotron_frequency(1.0, charge=-elementary_charge):.4g} rad/s")
    -1.759e+11 rad/s
    """
    return _out(
        np.asarray(charge, dtype=float) * np.abs(np.asarray(field, dtype=float)) / mass
    )


def thermal_speed(temperature, mass=electron_mass, convention="sqrt(T/m)"):
    """Compute the thermal speed of a Maxwellian of temperature T.

    Parameters
    ----------
    temperature : float or array_like
        Temperature T, in eV.
    mass : float or array_like, optional
        Particle mass m, in kg. Default: the electron mass.
    convention : {"sqrt(T/m)", "sqrt(2T/m)", "mean"}, optional
        ``"sqrt(T/m)"``: the standard deviation of each velocity component, as in the NRL
        formulary and in Struphy's Maxwellians and ``velocity_scale="thermal"``;
        ``"sqrt(2T/m)"``: the most probable speed, common in kinetic theory; ``"mean"``: the mean
        speed √(8T/(πm)). Default: ``"sqrt(T/m)"``.

    Returns
    -------
    float or numpy.ndarray
        The thermal speed, in m/s.

    Raises
    ------
    ValueError
        If ``convention`` is unknown.

    References
    ----------
    NRL Plasma Formulary: v_Te = √(k T_e / m_e) = 4.19e7 × √(T_e / eV) cm/s.

    Examples
    --------
    >>> print(f"{thermal_speed(1.0):.4g} m/s")
    4.194e+05 m/s
    >>> print(f"{thermal_speed(1.0, convention='sqrt(2T/m)'):.4g} m/s")
    5.931e+05 m/s
    """
    if convention not in _THERMAL_CONVENTIONS:
        raise ValueError(
            f"convention must be one of {list(_THERMAL_CONVENTIONS)}; got {convention!r}"
        )
    temperature = np.asarray(temperature, dtype=float)
    return _out(
        np.sqrt(_THERMAL_CONVENTIONS[convention] * temperature * ev_to_joule / mass)
    )


def debye_length(density, temperature):
    """Compute the electron Debye length λ_D = √(ε₀ T / (n e²)).

    Parameters
    ----------
    density : float or array_like
        Electron density n, in m⁻³.
    temperature : float or array_like
        Electron temperature T, in eV.

    Returns
    -------
    float or numpy.ndarray
        λ_D, in m. With the default :func:`thermal_speed`, λ_D = v_th / ω_pe.

    References
    ----------
    NRL Plasma Formulary: λ_D = 7.43e2 × √(T/eV) / √(n/cm⁻³) cm.

    Examples
    --------
    >>> print(f"{debye_length(1e6, 1.0):.4g} m")  # 1 cm⁻³, 1 eV
    7.434 m
    """
    density = np.asarray(density, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    return _out(
        np.sqrt(vacuum_permittivity * temperature * ev_to_joule / (density * _e**2))
    )


def larmor_radius(
    field,
    temperature=None,
    perpendicular_speed=None,
    mass=electron_mass,
    charge=elementary_charge,
):
    """Compute the Larmor (gyro-) radius ρ = m v⊥ / (|q| B).

    Give either the temperature, for the thermal Larmor radius with v⊥ = √(T/m), or the
    perpendicular speed.

    Parameters
    ----------
    field : float or array_like
        Magnetic field B, in T.
    temperature : float or array_like, optional
        Temperature T, in eV; then v⊥ = √(T/m) (the NRL convention).
    perpendicular_speed : float or array_like, optional
        The perpendicular speed v⊥, in m/s.
    mass : float or array_like, optional
        Particle mass m, in kg. Default: the electron mass.
    charge : float or array_like, optional
        Particle charge q, in C (its sign doesn't matter). Default: e.

    Returns
    -------
    float or numpy.ndarray
        ρ, in m.

    Raises
    ------
    ValueError
        If not exactly one of ``temperature`` and ``perpendicular_speed`` is given.

    References
    ----------
    NRL Plasma Formulary: r_e = 2.38 × √(T_e/eV) / (B/G) cm, r_i = 1.02e2 × √μ √(T_i/eV) / (Z B/G) cm.

    Examples
    --------
    >>> # 1 keV electron, 1 T
    >>> print(f"{larmor_radius(1.0, temperature=1e3):.4g} m")
    7.54e-05 m
    >>> radius = larmor_radius(2.0, perpendicular_speed=1e6, mass=proton_mass)
    >>> print(f"{radius:.4g} m")
    0.00522 m
    """
    if (temperature is None) == (perpendicular_speed is None):
        raise ValueError("give exactly one of temperature and perpendicular_speed")
    if perpendicular_speed is None:
        perpendicular_speed = thermal_speed(temperature, mass)
    speed = np.asarray(perpendicular_speed, dtype=float)
    return _out(
        mass * np.abs(speed) / (np.abs(charge) * np.abs(np.asarray(field, dtype=float)))
    )


def inertial_length(density, mass=electron_mass, charge=elementary_charge):
    """Compute the inertial length (skin depth) d = c / ω_p.

    Parameters
    ----------
    density : float or array_like
        Number density n of the species, in m⁻³.
    mass : float or array_like, optional
        Particle mass m, in kg. Default: the electron mass.
    charge : float or array_like, optional
        Particle charge q, in C. Default: e.

    Returns
    -------
    float or numpy.ndarray
        d, in m. For ions, d_i = v_A / Ω_i.

    References
    ----------
    NRL Plasma Formulary: c/ω_pe = 5.31e5 / √(n_e/cm⁻³) cm, c/ω_pi = 2.28e7 √μ / (Z √(n_i/cm⁻³)) cm.

    Examples
    --------
    >>> print(f"{inertial_length(1e20):.4g} m")  # electrons
    0.0005314 m
    >>> print(f"{inertial_length(1e20, mass=proton_mass):.4g} m")  # protons
    0.02277 m
    """
    return _out(speed_of_light / plasma_frequency(density, mass, charge))


def alfven_speed(field, density, mass_number=1, relativistic=False):
    """Compute the Alfvén speed v_A = B / √(μ₀ n m_i), with m_i = mass_number × m_p.

    Parameters
    ----------
    field : float or array_like
        Magnetic field B, in T.
    density : float or array_like
        Ion density n, in m⁻³ (the mass density is n m_i; electrons are neglected).
    mass_number : float or array_like, optional
        Ion mass in proton masses, as in Struphy. Default: ``1``.
    relativistic : bool, optional
        If true, return the relativistic v_A / √(1 + v_A²/c²). Default: ``False``.

    Returns
    -------
    float or numpy.ndarray
        v_A, in m/s.

    References
    ----------
    NRL Plasma Formulary: v_A = 2.18e11 × (B/G) / √(μ n_i/cm⁻³) cm/s.

    Examples
    --------
    >>> # deuterium, 1 T
    >>> print(f"{alfven_speed(1.0, 1e20, mass_number=2):.4g} m/s")
    1.542e+06 m/s
    """
    field = np.abs(np.asarray(field, dtype=float))
    density = np.asarray(density, dtype=float)
    speed = field / np.sqrt(vacuum_permeability * density * mass_number * proton_mass)
    if relativistic:
        speed = speed / np.sqrt(1 + (speed / speed_of_light) ** 2)
    return _out(speed)


def sound_speed(
    electron_temperature,
    ion_temperature=0.0,
    mass_number=1,
    charge_number=1,
    electron_gamma=1.0,
    ion_gamma=3.0,
):
    """Compute the ion sound speed c_s = √((γ_e Z T_e + γ_i T_i) / m_i).

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
    electron_gamma : float, optional
        Adiabatic index γ_e of the electrons. Default: ``1`` (isothermal electrons).
    ion_gamma : float, optional
        Adiabatic index γ_i of the ions. Default: ``3`` (1-D adiabatic ions).

    Returns
    -------
    float or numpy.ndarray
        c_s, in m/s.

    References
    ----------
    NRL Plasma Formulary: C_s = 9.79e5 × √(γ Z T_e / (μ eV)) cm/s.

    Chen, Introduction to Plasma Physics, section 4.9 (ion acoustic waves).

    Examples
    --------
    >>> print(f"{sound_speed(1.0):.4g} m/s")
    9787 m/s
    """
    te = np.asarray(electron_temperature, dtype=float)
    ti = np.asarray(ion_temperature, dtype=float)
    energy = (electron_gamma * charge_number * te + ion_gamma * ti) * ev_to_joule
    return _out(np.sqrt(energy / (np.asarray(mass_number, dtype=float) * proton_mass)))


def plasma_beta(density, temperature, field):
    """Compute the plasma beta β = n T / (B²/(2μ₀)), the ratio of thermal to magnetic pressure.

    For several species, add their β (or pass the summed pressure n T of all species).

    Parameters
    ----------
    density : float or array_like
        Number density n, in m⁻³.
    temperature : float or array_like
        Temperature T, in eV.
    field : float or array_like
        Magnetic field B, in T.

    Returns
    -------
    float or numpy.ndarray
        β (dimensionless).

    References
    ----------
    NRL Plasma Formulary: β = 8π n k T / B² = 4.03e-11 n T / B² (cgs, T in eV, B in G).

    Examples
    --------
    >>> print(f"{plasma_beta(1e20, 1e4, 5.0):.4f}")  # 10 keV, 5 T
    0.0161
    """
    density = np.asarray(density, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    field = np.asarray(field, dtype=float)
    return _out(
        2 * vacuum_permeability * density * temperature * ev_to_joule / field**2
    )


def plasma_parameter(density, temperature):
    """Compute the plasma parameter N_D = (4π/3) n λ_D³, the number of electrons in a Debye sphere.

    Parameters
    ----------
    density : float or array_like
        Electron density n, in m⁻³.
    temperature : float or array_like
        Electron temperature T, in eV.

    Returns
    -------
    float or numpy.ndarray
        N_D (dimensionless). A plasma is weakly coupled for N_D ≫ 1.

    References
    ----------
    NRL Plasma Formulary: (4π/3) n λ_D³ = 1.72e9 × (T/eV)^(3/2) / √(n/cm⁻³).

    Examples
    --------
    >>> print(f"{plasma_parameter(1e6, 1.0):.3g}")  # 1 cm⁻³, 1 eV
    1.72e+09
    """
    density = np.asarray(density, dtype=float)
    return _out(4 * np.pi / 3 * density * debye_length(density, temperature) ** 3)


# Braginskii's α₀ (η∥/η⊥) against 1/Z, for Z = ∞, 4, 3, 2, 1
_ALPHA0_INVERSE_Z = np.array([0.0, 0.25, 1 / 3, 0.5, 1.0])
_ALPHA0 = np.array([0.2949, 0.3752, 0.3965, 0.4408, 0.5129])


def lower_hybrid_frequency(density, field, mass_number=1, charge_number=1):
    """Compute the lower hybrid frequency, 1/ω_LH² = 1/(Ω_i² + ω_pi²) + 1/(|Ω_e| Ω_i).

    Parameters
    ----------
    density : float or array_like
        Electron density n_e, in m⁻³ (the ion density is n_e / Z).
    field : float or array_like
        Magnetic field B, in T.
    mass_number : float or array_like, optional
        Ion mass in proton masses. Default: ``1``.
    charge_number : float or array_like, optional
        Ion charge number Z. Default: ``1``.

    Returns
    -------
    float or numpy.ndarray
        ω_LH, in rad/s. For ω_pi ≫ Ω_i it tends to √(|Ω_e| Ω_i).

    References
    ----------
    Chen, Introduction to Plasma Physics, section 4.11 (lower hybrid frequency).

    NRL Plasma Formulary: ω_LH = [(Ω_i Ω_e)⁻¹ + ω_pi⁻²]^(−1/2).

    Examples
    --------
    >>> print(f"{lower_hybrid_frequency(1e20, 1.0) / (2 * np.pi):.4g} Hz")
    6.237e+08 Hz
    """
    density = np.asarray(density, dtype=float)
    ion_mass = np.asarray(mass_number, dtype=float) * proton_mass
    ion_charge = np.asarray(charge_number, dtype=float) * _e
    omega_ci = cyclotron_frequency(field, ion_mass, ion_charge)
    omega_ce = cyclotron_frequency(field)
    omega_pi2 = plasma_frequency(density / charge_number, ion_mass, ion_charge) ** 2
    return _out(1 / np.sqrt(1 / (omega_ci**2 + omega_pi2) + 1 / (omega_ce * omega_ci)))


def upper_hybrid_frequency(density, field):
    """Compute the upper hybrid frequency ω_UH = √(ω_pe² + Ω_e²).

    Parameters
    ----------
    density : float or array_like
        Electron density n_e, in m⁻³.
    field : float or array_like
        Magnetic field B, in T.

    Returns
    -------
    float or numpy.ndarray
        ω_UH, in rad/s.

    References
    ----------
    Chen, Introduction to Plasma Physics, section 4.10 (upper hybrid frequency).

    Examples
    --------
    >>> print(f"{upper_hybrid_frequency(1e19, 1.0):.4g} rad/s")
    2.505e+11 rad/s
    """
    return _out(
        np.sqrt(plasma_frequency(density) ** 2 + cyclotron_frequency(field) ** 2)
    )


def struphy_units(
    x=1.0,
    B=1.0,
    n=1.0,
    kBT=None,
    velocity_scale="light",
    mass_number=None,
    charge_number=None,
):
    """Compute the units of Struphy's normalization from its base units.

    Mirrors ``struphy.physics.physics.Units.derive_units`` (the base units are those of
    ``struphy.io.options.BaseUnits`` in a parameter file, the velocity scale is the model's
    ``velocity_scale`` and the mass and charge numbers are those of the model's bulk species):

    * velocity v: c for ``"light"``; B / √(A m_p n μ₀) for ``"alfvén"``; (Z e B / (A m_p)) x for
      ``"cyclotron"``; √(kBT / (A m_p)) for ``"thermal"``; 1 m/s for ``None``,
    * time t = x / v, and the frequency 1/t (normalized angular frequencies ω t are in rad/t),
    * with a bulk species (``mass_number`` given): pressure p = A m_p n v² (= B²/μ₀ for
      ``"alfvén"``), mass density ρ = A m_p n, current density j = e n v.

    Parameters
    ----------
    x : float, optional
        Unit of length, in m (``BaseUnits.x``). Default: ``1``.
    B : float, optional
        Unit of magnetic field, in T (``BaseUnits.B``). Default: ``1``.
    n : float, optional
        Unit of number density, in 1e20 m⁻³ as in ``BaseUnits.n`` (not in m⁻³). Default: ``1``.
    kBT : float, optional
        Unit of thermal energy, in keV as in ``BaseUnits.kBT``; needed for ``"thermal"``.
    velocity_scale : {"light", "alfvén", "cyclotron", "thermal", None}, optional
        The model's velocity scale (``"alfven"`` is accepted too). Default: ``"light"``.
    mass_number : float, optional
        Mass number A of the bulk species, in proton masses; needed except for ``"light"`` and
        ``None``.
    charge_number : float, optional
        Charge number Z of the bulk species; needed for ``"cyclotron"``.

    Returns
    -------
    dict
        Floats, in SI units except ``kBT``: ``x`` (m), ``B`` (T), ``n`` (m⁻³), ``kBT`` (keV or
        None), ``v`` (m/s), ``t`` (s), ``frequency`` (1/s), ``p`` (Pa), ``rho`` (kg/m³) and ``j``
        (A/m²); the last three are None without a bulk species. ``STRUPHY_UNIT_SYMBOLS`` holds
        these unit names.

    Raises
    ------
    ValueError
        If ``velocity_scale`` is unknown or a quantity it needs is missing.

    References
    ----------
    Struphy source: ``struphy/physics/physics.py`` (``Units``, ``ConstantsOfNature``) and
    ``struphy/io/options.py`` (``BaseUnits``); the "normalization" page of Struphy's docs.

    Examples
    --------
    >>> units = struphy_units(
    ...     x=1.0, B=1.0, n=1.0, velocity_scale="alfvén", mass_number=2
    ... )
    >>> print(f"v = {units['v']:.4g} m/s, t = {units['t']:.4g} s")
    v = 1.542e+06 m/s, t = 6.484e-07 s
    >>> print(f"p = {units['p']:.4g} Pa")  # B²/μ₀
    p = 7.958e+05 Pa
    """
    density = n * 1e20
    scale = "alfvén" if velocity_scale == "alfven" else velocity_scale
    if scale not in ("light", "alfvén", "cyclotron", "thermal", None):
        raise ValueError(f"unknown velocity_scale {velocity_scale!r}")
    if scale in ("alfvén", "cyclotron", "thermal") and mass_number is None:
        raise ValueError(
            f'velocity_scale "{scale}" needs the mass_number of the bulk species'
        )
    if scale == "cyclotron" and charge_number is None:
        raise ValueError(
            'velocity_scale "cyclotron" needs the charge_number of the bulk species'
        )
    if scale == "thermal" and kBT is None:
        raise ValueError('velocity_scale "thermal" needs kBT')
    if scale is None:
        v = 1.0
    elif scale == "light":
        v = speed_of_light
    elif scale == "alfvén":
        v = B / np.sqrt(density * mass_number * proton_mass * vacuum_permeability)
    elif scale == "cyclotron":
        v = charge_number * _e * B / (mass_number * proton_mass) * x
    else:
        v = np.sqrt(kBT * 1000 * _e / (proton_mass * mass_number))
    v = float(v)
    t = x / v
    units = {
        "x": float(x),
        "B": float(B),
        "n": float(density),
        "kBT": kBT,
        "v": v,
        "t": t,
        "frequency": 1 / t,
        "p": None,
        "rho": None,
        "j": None,
    }
    if mass_number is not None:
        units["p"] = float(mass_number * proton_mass * density * v**2)
        units["rho"] = float(mass_number * proton_mass * density)
        units["j"] = float(_e * density * v)
    return units


def struphy_equation_parameters(units, charge_number=1, mass_number=1):
    """Compute the equation parameters α, ε and κ of one Struphy species.

    Mirrors ``struphy.models.species.Species.EquationParameters``, with ω_p and Ω_c of the
    species at the unit density and unit field: α = ω_p/Ω_c, ε = 1/(Ω_c t) and κ = ω_p t.

    Parameters
    ----------
    units : dict
        The units from :func:`struphy_units`.
    charge_number : float, optional
        Charge number Z of the species. Default: ``1``.
    mass_number : float, optional
        Mass number A of the species, in proton masses. Default: ``1``.

    Returns
    -------
    dict
        ``alpha``, ``epsilon`` and ``kappa``.

    References
    ----------
    Struphy source: ``struphy/models/species.py`` (``Species.EquationParameters``).

    Examples
    --------
    >>> units = struphy_units(velocity_scale="alfvén", mass_number=1)
    >>> params = struphy_equation_parameters(units)
    >>> print({key: f"{value:.4g}" for key, value in params.items()})
    {'alpha': '137.4', 'epsilon': '0.02277', 'kappa': '6036'}
    """
    mass = mass_number * proton_mass
    charge = charge_number * _e
    omega_p = float(plasma_frequency(units["n"], mass, charge))
    omega_c = float(charge * units["B"] / mass)
    return {
        "alpha": omega_p / omega_c,
        "epsilon": 1 / (omega_c * units["t"]),
        "kappa": omega_p * units["t"],
    }
