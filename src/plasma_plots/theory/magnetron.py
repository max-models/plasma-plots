"""Crossed-field (magnetron) discharges: Hull cutoff, Brillouin flow, drifts and gradient-drift modes.

SI units throughout (volts, tesla, metres, seconds, m⁻³); scalars or broadcasting numpy arrays.
Electrons are the default species of the magnetic-confinement functions.

Gradient-drift sign convention
------------------------------
The density profile has n'/n = 1/L_n (L_n signed) and the equilibrium field E₀ is along the same
axis, so the E×B drift is v_E = E₀/B along the azimuthal axis y, and k is the azimuthal wavenumber
(modes ~ exp(i(k y − ω t))). With this convention E₀ L_n > 0 is the unstable (Simon–Hoh) case:
the field and the density gradient point the same way. For E₀ L_n < 0 the modes are stable.

References
----------
Hull cutoff: A. W. Hull, Phys. Rev. 18, 31 (1921); Lieberman and Lichtenberg, Principles of
Plasma Discharges and Materials Processing, 2nd ed., section 12.4 (not re-read here; the
formulas below are derived from energy and canonical angular momentum conservation).

A. Simon, Phys. Fluids 6, 382 (1963); F. C. Hoh, Phys. Fluids 6, 1184 (1963).

Direct evidence of the gradient drift instability being the origin of a rotating spoke in a
crossed field plasma, arXiv:2103.06559, Eq. (1) (the dispersion relation of Kawashima, Lakhin
and co-workers' Hall-thruster work, unmagnetized motionless-ion response).

H. Alfvén, On the Origin of the Solar System (Oxford, 1954): critical ionisation velocity.

Examples
--------
The Hull cutoff field of a planar magnetron with a 5 cm gap at 300 V:

>>> print(f"{hull_cutoff(voltage=300.0, gap=0.05) * 1e3:.4f} mT")
1.1681 mT
"""

from __future__ import annotations

import numpy as np

from plasma_plots.theory import kinetic, orbits, parameters
from plasma_plots.theory.parameters import (
    electron_mass,
    elementary_charge,
    proton_mass,
    vacuum_permittivity,
)


def _out(value):
    return np.asarray(value, dtype=float)[()]


# ---------------------------------------------------------------------------------------------
# Confinement
# ---------------------------------------------------------------------------------------------
def hull_cutoff(
    voltage=None,
    field=None,
    *,
    gap=None,
    anode_radius=None,
    cathode_radius=None,
    mass=electron_mass,
    charge=elementary_charge,
):
    """Compute the Hull cutoff of a planar or coaxial magnetron (give all but one of voltage/field).

    A particle starting at rest on the cathode in the crossed fields just grazes the anode when
    ``q B G = sqrt(2 m q V)``, i.e. ``B_c = sqrt(2 m V / q) / G`` and ``V_c = q B² G² / (2 m)``.
    The geometry factor is G = d for a planar gap d, and for coaxial cylinders (axial B) the
    conserved canonical angular momentum gives G = |r_a² − r_c²| / (2 r_a), with r_a the anode
    and r_c the cathode radius. For r_a/r_c → 1, G → d = |r_a − r_c| (the planar result).
    Non-relativistic.

    Parameters
    ----------
    voltage : float or array_like, optional
        Anode–cathode voltage V, in V. Omit it to compute the critical voltage.
    field : float or array_like, optional
        Magnetic field B, in T. Omit it to compute the critical field.
    gap : float or array_like, optional
        Planar gap d, in m. Give this or both radii.
    anode_radius, cathode_radius : float or array_like, optional
        Coaxial geometry, in m.
    mass, charge : float or array_like, optional
        Particle mass (kg) and charge magnitude (C). Default: electron.

    Returns
    -------
    float or numpy.ndarray
        B_c in T if ``field`` is omitted, otherwise V_c in V. Above B_c (or below V_c) the
        particles cannot reach the anode.

    Raises
    ------
    ValueError
        If not exactly one of voltage and field is omitted, or the geometry is not given.

    References
    ----------
    Hull (1921); Lieberman and Lichtenberg, section 12.4.

    Examples
    --------
    >>> print(f"{hull_cutoff(voltage=300.0, gap=0.05) * 1e3:.4f} mT")
    1.1681 mT
    >>> print(
    ...     f"{hull_cutoff(voltage=300.0, anode_radius=0.06, cathode_radius=0.01) * 1e3:.3f} mT"
    ... )
    2.003 mT
    >>> print(f"{hull_cutoff(field=1.16814e-3, gap=0.05):.1f} V")
    300.0 V
    """
    if (voltage is None) == (field is None):
        raise ValueError("give exactly one of voltage and field")
    if gap is not None:
        factor = np.asarray(gap, dtype=float)
    elif anode_radius is not None and cathode_radius is not None:
        ra = np.asarray(anode_radius, dtype=float)
        rc = np.asarray(cathode_radius, dtype=float)
        factor = np.abs(ra**2 - rc**2) / (2 * ra)
    else:
        raise ValueError("give gap, or both anode_radius and cathode_radius")
    charge = np.abs(np.asarray(charge, dtype=float))
    if field is None:
        return _out(np.sqrt(2 * mass * np.asarray(voltage, dtype=float) / charge) / factor)
    return _out(charge * np.asarray(field, dtype=float) ** 2 * factor**2 / (2 * mass))


def brillouin_flow(field, radius=None, mass=electron_mass, charge=elementary_charge):
    """Compute Brillouin flow: the rigidly rotating, uniform-density limit of a non-neutral cloud.

    Radial force balance of a cold non-neutral species rotating at ω_θ gives
    ω_p² = 2 ω_θ (Ω − ω_θ) with Ω = qB/m; the maximum density for which the rotation is rigid is
    ω_θ = Ω/2 and ω_p² = Ω²/2, i.e. n_B = ε₀ B² / (2 m).

    Parameters
    ----------
    field : float or array_like
        Magnetic field B, in T.
    radius : float or array_like, optional
        If given, the azimuthal speed v_θ = ω_θ r at this radius is returned as well.
    mass, charge : float or array_like, optional
        Species mass (kg) and charge magnitude (C). Default: electron.

    Returns
    -------
    dict
        ``"density"`` (m⁻³), ``"angular_frequency"`` (rad/s) and, if ``radius`` is given,
        ``"speed"`` (m/s).

    References
    ----------
    Brillouin, Phys. Rev. 67, 260 (1945); Davidson, Physics of Nonneutral Plasmas (1990), ch. 3.

    Examples
    --------
    >>> flow = brillouin_flow(0.05, radius=0.02)
    >>> print(f"{flow['density']:.4g} m^-3, {flow['speed']:.4g} m/s")
    1.215e+16 m^-3, 8.794e+07 m/s
    """
    field = np.abs(np.asarray(field, dtype=float))
    charge = np.abs(np.asarray(charge, dtype=float))
    omega = charge * field / mass / 2
    result = {
        "density": _out(vacuum_permittivity * field**2 / (2 * mass)),
        "angular_frequency": _out(omega),
    }
    if radius is not None:
        result["speed"] = _out(omega * np.asarray(radius, dtype=float))
    return result


def hall_parameter(field, collision_frequency, mass=electron_mass, charge=elementary_charge):
    """Compute the Hall parameter ω_c / ν = |q| B / (m ν).

    Parameters
    ----------
    field : float or array_like
        Magnetic field B, in T.
    collision_frequency : float or array_like
        Momentum-transfer collision frequency ν, in 1/s.
    mass, charge : float or array_like, optional
        Species mass (kg) and charge (C). Default: electron.

    Returns
    -------
    float or numpy.ndarray
        ω_c τ, dimensionless (positive). Classical cross-field transport is reduced by
        1/(1 + (ω_c/ν)²); Bohm diffusion corresponds to an effective value of 16.

    References
    ----------
    Chen, Introduction to Plasma Physics, section 5.5.

    Examples
    --------
    >>> print(f"{hall_parameter(0.05, 1e8):.4g}")
    87.94
    """
    omega_c = np.abs(parameters.cyclotron_frequency(field, mass, charge))
    return _out(omega_c / np.asarray(collision_frequency, dtype=float))


def critical_ionisation_velocity(voltage, mass_number=1.0):
    """Compute Alfvén's critical ionisation velocity v_c = sqrt(2 e U / m).

    The speed at which a neutral of mass m has the kinetic energy e U. Taken with U the
    ionisation potential (Alfvén's original hypothesis) or with U the discharge voltage when
    comparing to a spoke speed (Hecimovic and von Keudell 2018).

    Parameters
    ----------
    voltage : float or array_like
        U, in V.
    mass_number : float or array_like, optional
        Neutral mass in proton masses (as in :func:`~plasma_plots.theory.parameters.alfven_speed`),
        e.g. 39.948 for argon. Default: ``1``.

    Returns
    -------
    float or numpy.ndarray
        v_c, in m/s.

    References
    ----------
    Alfvén (1954); Hecimovic and von Keudell, J. Phys. D 51, 453001 (2018).

    Examples
    --------
    Argon at 100 V (equal to sqrt(2 e U / m) = 21.9 km/s):

    >>> print(f"{critical_ionisation_velocity(100.0, 39.948):.1f} m/s")
    21899.0 m/s
    """
    mass = np.asarray(mass_number, dtype=float) * proton_mass
    return _out(np.sqrt(2 * elementary_charge * np.asarray(voltage, dtype=float) / mass))


def exb_drift_speed(electric_field, field):
    """Compute the E×B drift speed E/B for perpendicular fields (via :func:`orbits.exb_drift`).

    Parameters
    ----------
    electric_field : float or array_like
        Component E⊥ of E perpendicular to B, in V/m (signed).
    field : float or array_like
        Magnetic field B, in T.

    Returns
    -------
    float or numpy.ndarray
        |E⊥| / B, in m/s.

    Examples
    --------
    >>> print(f"{exb_drift_speed(1e4, 0.05):.4g} m/s")
    2e+05 m/s
    """
    E, B = np.broadcast_arrays(np.asarray(electric_field, dtype=float), np.asarray(field, dtype=float))
    zero = np.zeros_like(E)
    drift = orbits.exb_drift(np.stack([E, zero, zero], axis=-1), np.stack([zero, zero, B], axis=-1))
    return _out(np.linalg.norm(drift, axis=-1))


# ---------------------------------------------------------------------------------------------
# Gradient-drift (Simon-Hoh) modes
# ---------------------------------------------------------------------------------------------
def _ion_cyclotron(field, mass_number, charge_number):
    return parameters.cyclotron_frequency(
        field,
        np.asarray(mass_number, dtype=float) * proton_mass,
        np.asarray(charge_number) * elementary_charge,
    )


def simon_hoh_frequency(
    wavenumber,
    electric_field,
    field,
    gradient_length,
    mass_number=1.0,
    charge_number=1.0,
):
    """Compute the two roots of the cold-electron Simon–Hoh dispersion relation.

    Assumptions: collisionless, k along E₀×B (k_∥ = 0), electrostatic, quasi-neutral; electrons
    magnetized with E×B and diamagnetic drifts and no Larmor-radius correction; ions unmagnetized
    (ω ≫ Ω_i) and cold, with inertia, so the ion density response is n_i1/n₀ = k² c_s² φ/ω² in
    units of e/T_e. The electron continuity n₁/n₀ = −(k φ / B L_n) / (ω − ω_E) closes to

        ω² = a (ω − ω_E),   a = Ω_i k L_n,   ω_E = k E₀ / B,   Ω_i = Z e B / m_i,

    independent of T_e (the sound-speed and diamagnetic terms cancel). The roots are
    ω = [a ± sqrt(a² − 4 a ω_E)] / 2: unstable for ω_E / a > 1/4, i.e. E₀/(B Ω_i L_n) > 1/4, with
    real part a/2 (phase velocity Ω_i L_n / 2, independent of E₀ and k) and growth rate
    sqrt(a ω_E − a²/4), which is k sqrt(Ω_i L_n v_E) for v_E ≫ Ω_i L_n. L_n and E₀ are signed
    as in the module docstring (E₀ L_n > 0 unstable). This is the ω_ce → ∞, ν → 0 limit of
    :func:`gradient_drift_frequency`.

    Parameters
    ----------
    wavenumber : float or array_like
        Azimuthal wavenumber k, in 1/m.
    electric_field : float or array_like
        E₀, in V/m (signed).
    field : float or array_like
        B, in T.
    gradient_length : float or array_like
        Signed density gradient length L_n = n/(dn/dx), in m.
    mass_number, charge_number : float or array_like, optional
        Ion mass (proton masses) and charge number. Default: ``1``.

    Returns
    -------
    numpy.ndarray
        Complex ω (rad/s) of the root with the larger imaginary part. The phase velocity is
        ``.real / k``.

    References
    ----------
    Simon, Phys. Fluids 6, 382 (1963); Hoh, Phys. Fluids 6, 1184 (1963); arXiv:2103.06559 Eq. (1)
    (limit; re-derived here).

    Examples
    --------
    Argon, E₀ = 10 kV/m, B = 50 mT, L_n = 1 cm, k = 2π/(5 cm):

    >>> k = 2 * np.pi / 0.05
    >>> w = simon_hoh_frequency(k, 1e4, 0.05, 0.01, 39.948)
    >>> print(f"{w.real / k:.1f} m/s, {w.imag:.4g} 1/s")
    599.5 m/s, 1.944e+06 1/s
    """
    k = np.asarray(wavenumber, dtype=float)
    a = _ion_cyclotron(field, mass_number, charge_number) * k * np.asarray(gradient_length, dtype=float)
    omega_e = k * np.asarray(electric_field, dtype=float) / np.asarray(field, dtype=float)
    root = np.sqrt((a * a - 4 * a * omega_e).astype(complex))
    first, second = (a + root) / 2, (a - root) / 2
    return np.where(first.imag >= second.imag, first, second)[()]


def simon_hoh_growth(
    wavenumber,
    electric_field,
    field,
    gradient_length,
    mass_number=1.0,
    charge_number=1.0,
):
    """Compute the Simon–Hoh growth rate γ = Im ω (zero if stable); see :func:`simon_hoh_frequency`.

    Returns
    -------
    float or numpy.ndarray
        γ ≥ 0, in 1/s.

    Examples
    --------
    >>> print(
    ...     f"{simon_hoh_growth(2 * np.pi / 0.05, 1e4, 0.05, 0.01, 39.948):.4g} 1/s"
    ... )
    1.944e+06 1/s
    >>> print(simon_hoh_growth(2 * np.pi / 0.05, 1e4, 0.05, -0.01, 39.948))
    0.0
    """
    omega = simon_hoh_frequency(wavenumber, electric_field, field, gradient_length, mass_number, charge_number)
    return _out(np.maximum(np.imag(omega), 0.0))


def gradient_drift_frequency(
    wavenumber,
    electric_field,
    field,
    gradient_length,
    electron_temperature,
    mass_number=1.0,
    collision_frequency=0.0,
):
    """Compute the gradient-drift dispersion relation with finite electron Larmor radius.

    Solves, for the root with the largest imaginary part, Eq. (1) of arXiv:2103.06559 (the form
    used for Hall-thruster and magnetron spokes: unmagnetized ions, magnetized electrons with
    Larmor radius ρ_e and electron–neutral collisions ν):

        [ω_d + k²ρ_e² (ω − ω_E + iν)] ω² = k² c_s² [(ω − ω_E) + k²ρ_e² (ω − ω_E + iν)],

    with ω_E = k E₀/B, ω_d = k T_e/(e B L_n), ρ_e = sqrt(T_e/m_e)/ω_ce and c_s = sqrt(T_e/m_i) (a
    cubic in ω, solved with :func:`numpy.roots`). Sign convention as in the module docstring.
    The ρ_e and ν terms are taken from that paper and not re-derived here; for ν = 0 and k²ρ_e² ω_E/ω_d → 0
    (ω_ce → ∞, at fixed T_e: a 0.4 % effect in the doctest) the relation reduces to :func:`simon_hoh_frequency` (checked in the tests).

    Parameters
    ----------
    wavenumber : float or array_like
        Azimuthal wavenumber k, in 1/m.
    electric_field : float or array_like
        E₀, in V/m (signed).
    field : float or array_like
        B, in T.
    gradient_length : float or array_like
        Signed density gradient length L_n, in m.
    electron_temperature : float or array_like
        T_e, in eV (also the temperature in c_s; cold ions).
    mass_number : float or array_like, optional
        Ion mass in proton masses. Default: ``1``.
    collision_frequency : float or array_like, optional
        Electron–neutral collision frequency ν, in 1/s. Default: ``0``.

    Returns
    -------
    numpy.ndarray
        Complex ω, in rad/s.

    Examples
    --------
    >>> k = 2 * np.pi / 0.05
    >>> w = gradient_drift_frequency(k, 1e4, 0.05, 0.01, 3.0, 39.948)
    >>> print(f"{w.real / k:.1f} m/s, {w.imag:.4g} 1/s")
    603.8 m/s, 1.948e+06 1/s
    """
    args = (
        wavenumber,
        electric_field,
        field,
        gradient_length,
        electron_temperature,
        mass_number,
        collision_frequency,
    )
    k, E0, B, L, Te, A, nu = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in args))
    te = Te * elementary_charge
    wce = elementary_charge * np.abs(B) / electron_mass
    kappa = k**2 * (te / electron_mass) / wce**2
    cs2k2 = k**2 * te / (A * proton_mass)
    wE = k * E0 / B
    wd = k * te / (elementary_charge * B * L)
    out = np.empty(k.shape, dtype=complex)
    for idx in np.ndindex(k.shape):
        kap = kappa[idx]
        coeffs = [
            kap,
            wd[idx] - kap * wE[idx] + 1j * kap * nu[idx],
            -cs2k2[idx] * (1 + kap),
            cs2k2[idx] * ((1 + kap) * wE[idx] - 1j * kap * nu[idx]),
        ]
        roots = np.roots(coeffs if kap != 0 else coeffs[1:])
        out[idx] = roots[np.argmax(roots.imag)]
    return out[()]


def gradient_drift_growth(*args, **kwargs):
    """Compute the gradient-drift growth rate γ = Im ω (zero if stable).

    Takes the arguments of :func:`gradient_drift_frequency`.

    Returns
    -------
    float or numpy.ndarray
        γ ≥ 0, in 1/s.

    Examples
    --------
    >>> k = 2 * np.pi / 0.05
    >>> print(f"{gradient_drift_growth(k, 1e4, 0.05, 0.01, 3.0, 39.948):.4g} 1/s")
    1.948e+06 1/s
    """
    return _out(np.maximum(np.imag(gradient_drift_frequency(*args, **kwargs)), 0.0))


def gradient_drift_phase_velocity(*args, **kwargs):
    """Compute the phase velocity Re ω / k of the most unstable gradient-drift root.

    Takes the arguments of :func:`gradient_drift_frequency` (``wavenumber`` first). In the
    cold-electron limit it is Ω_i L_n / 2, far below the E×B drift for typical magnetrons.

    Returns
    -------
    float or numpy.ndarray
        Re ω / k, in m/s (signed with the convention of the module docstring).

    Examples
    --------
    >>> k = 2 * np.pi / 0.05
    >>> print(
    ...     f"{gradient_drift_phase_velocity(k, 1e4, 0.05, 0.01, 3.0, 39.948):.1f} m/s"
    ... )
    603.8 m/s
    """
    omega = gradient_drift_frequency(*args, **kwargs)
    return _out(np.real(omega) / np.asarray(args[0], dtype=float))


# ---------------------------------------------------------------------------------------------
# Lower-hybrid drift modes
# ---------------------------------------------------------------------------------------------
def lower_hybrid_drift(
    wavenumber,
    field,
    density,
    electron_temperature,
    ion_temperature,
    ion_drift,
    electron_drift,
    mass_number=1.0,
    parallel_ratio=0.0,
    diamagnetic_drift=0.0,
):
    """Compute the electrostatic lower-hybrid drift / modified two-stream root (Malkov and Sotnikov, Eq. 1).

    Local, electrostatic, collisionless; unmagnetized Maxwellian ions of charge e (ω ≫ Ω_i and
    k ρ_i ≫ 1); magnetized Maxwellian electrons with finite Larmor radius; k ⊥ B up to a small
    k_z ≪ k. With exp(i(k y − ω t)) the relation is

        1 + (2 ω_pi²/(k² v_Ti²)) [1 + ξ Z(ξ)] + (ω_pe²/Ω_ce²) (1 − Γ₀(b))/b
          − (k_z²/k²) ω_pe²/(ω − k V_e)² + (2 ω_pe²/(k² v_Te²)) k V_Δ/(ω − k V_e) = 0,

    with ξ = (ω − k V_i)/(|k| v_Ti), v_T = √(2T/m), Γ₀(b) = exp(−b) I₀(b) and
    b = k² T_e/(m_e Ω_ce²) = k² ρ_e². V_i and V_e are the ion and electron drifts along k (the
    instability depends on V_e − V_i, so V_i = 0 gives the ion rest frame) and V_Δ is the electron
    drift due to density, field and temperature gradients, taken as an input (the paper's
    V_Δ = −(v_Te²/Ω_ce) Γ₀(b) {κ_n − …} includes the Larmor-radius factors). In the source the
    thermal speed v_T and b are printed ambiguously (b = |k| v_Te/Ω_ce); the values above are
    those for which the electron response reduces to the Debye one, (1 − Γ₀)/(k² λ_De²), and
    the V_Δ term to the standard Maxwellian drift term; the cold-electron limit is checked in the
    tests.

    * With k_z = 0 and V_Δ = 0 every root is real: a drive is needed.
    * With k_z ≠ 0, V_Δ = 0 and cold ions the relation is the modified two-stream relation
      ω_pi²/(ω − k V_i)² + (k_z/k)² ω_pe²/(ω − k V_e)² = R, R = 1 + (ω_pe²/Ω_ce²)(1 − Γ₀)/b,
      which has a complex pair iff R k² (V_e − V_i)² < (ω_pi^{2/3} + (k_z/k)^{2/3} ω_pe^{2/3})³
      (Eq. (7) of the source, restated here and checked in the tests).
    * With k_z = 0 and V_Δ ≠ 0 the relation is the electrostatic lower-hybrid drift instability
      (LHDI): a lower-hybrid branch with ω of order ω_LH = ω_pi/√(1 + ω_pe²/Ω_ce²), driven by the
      gradient drift V_Δ through resonance with ions of relative drift V_i of a few v_Ti and
      T_i ~ T_e (the sign of V_Δ opposite to V_e − V_i, as in the example). This is the weak-β,
      electrostatic limit only: no magnetic fluctuations, no β-dependent stabilization, and no
      density-gradient (k_x) structure. The relation of this source was checked here against its
      cold-ion closed form and against scipy (see the tests), not against the original paper of
      Davidson and Gladd.

    The root with the largest growth rate is found by Newton iteration from several seeds with
    :func:`~plasma_plots.theory.kinetic.solve_dispersion`; ``nan`` if none converged.

    Parameters
    ----------
    wavenumber : float or array_like
        k along the drift, in 1/m.
    field : float or array_like
        B, in T.
    density : float or array_like
        n_e = n_i, in m⁻³.
    electron_temperature, ion_temperature : float or array_like
        T_e and T_i, in eV.
    ion_drift, electron_drift : float or array_like
        V_i and V_e along k, in m/s.
    mass_number : float or array_like, optional
        Ion mass in proton masses. Default: ``1``.
    parallel_ratio : float or array_like, optional
        k_z/k. Default: ``0``.
    diamagnetic_drift : float or array_like, optional
        V_Δ in m/s along k. Default: ``0``.

    Returns
    -------
    numpy.ndarray
        Complex ω in rad/s, in the frame in which V_i and V_e are given.

    References
    ----------
    M. A. Malkov and V. I. Sotnikov, arXiv:1805.01933, Eq. (1) (read in full; the equation there
    is adapted from R. C. Davidson and N. T. Gladd, Phys. Fluids 18, 1327 (1975), not read) and
    Eq. (7).

    Examples
    --------
    Xenon, n = 1e17 m⁻³, B = 20 mT, T_e = T_i = 10 eV, k ρ_e = 0.5 (k = 1326 1/m), ions drifting at
    −10 km/s (2.6 v_Ti) through electrons with V_Δ = 3 km/s and no E×B drift:

    >>> w = lower_hybrid_drift(
    ...     1326.0, 0.02, 1e17, 10.0, 10.0, -1e4, 0.0, 131.293, 0.0, 3e3
    ... )
    >>> print(f"{w.real:.4g} rad/s, {w.imag:.4g} 1/s")
    -8.989e+06 rad/s, 4.691e+06 1/s
    """
    k, B, n, Te, Ti, Vi, Ve, A, kz, Vd = np.broadcast_arrays(
        *(
            np.asarray(x, dtype=float)
            for x in (
                wavenumber,
                field,
                density,
                electron_temperature,
                ion_temperature,
                ion_drift,
                electron_drift,
                mass_number,
                parallel_ratio,
                diamagnetic_drift,
            )
        )
    )
    shape = k.shape
    k, B, n, Te, Ti, Vi, Ve, A, kz, Vd = (x.ravel() for x in (k, B, n, Te, Ti, Vi, Ve, A, kz, Vd))
    ion_mass = A * proton_mass
    wpe2 = n * elementary_charge**2 / (vacuum_permittivity * electron_mass)
    wpi2 = n * elementary_charge**2 / (vacuum_permittivity * ion_mass)
    wce = elementary_charge * np.abs(B) / electron_mass
    vte2 = 2 * Te * elementary_charge / electron_mass
    vti = np.sqrt(2 * Ti * elementary_charge / ion_mass)
    b = k**2 * vte2 / (2 * wce**2)
    with np.errstate(over="ignore"):
        gamma0 = np.where(
            b < 500,
            np.i0(np.minimum(b, 500.0)) * np.exp(-np.minimum(b, 500.0)),
            (1 + 1 / (8 * b) + 9 / (128 * b**2)) / np.sqrt(2 * np.pi * b),
        )
    flr = np.where(b < 1e-6, 1 - 0.75 * b, (1 - gamma0) / np.where(b == 0, 1.0, b))
    static = 1 + wpe2 / wce**2 * flr
    parallel = kz**2 * wpe2
    drive = 2 * wpe2 / (k**2 * vte2) * k * Vd
    ion_scale = 2 * wpi2 / (k * vti) ** 2

    def dielectric(omega, kk):
        w, _ = kinetic._response((omega - kk * Vi) / (np.abs(kk) * vti))
        shifted = omega - kk * Ve
        return static + ion_scale * w - parallel / shifted**2 + drive / shifted

    def slope(omega, kk):
        _, dw = kinetic._response((omega - kk * Vi) / (np.abs(kk) * vti))
        shifted = omega - kk * Ve
        return ion_scale * dw / (np.abs(kk) * vti) + 2 * parallel / shifted**3 - drive / shifted**2

    wlh = np.sqrt(wpi2 / (1 + wpe2 / wce**2))
    scale = np.abs(k * (Ve - Vi)) + np.abs(k * Vd) + wlh
    seeds = [
        k * Vi + f * k * (Ve - Vi) + 1j * g * scale for f in (0.05, 0.25, 0.5, 0.75, 0.95) for g in (0.02, 0.1, 0.3)
    ]
    seeds += [k * Vi + s * wlh + 1j * g * wlh for s in (-1.0, -0.5, 0.5, 1.0) for g in (0.05, 0.3)]
    # the roots of the cold-ion polynomial (the quartic of Eq. (2) with the drive term) as seeds
    cold = np.empty((4, k.size), dtype=complex)
    for i in range(k.size):
        ui, ue = np.poly1d([1.0, -k[i] * Vi[i]]), np.poly1d([1.0, -k[i] * Ve[i]])
        scalar = np.poly1d
        poly = (
            scalar([static[i]]) * ui**2 * ue**2
            - scalar([wpi2[i]]) * ue**2
            - scalar([parallel[i]]) * ui**2
            + scalar([drive[i]]) * ue * ui**2
        )
        found = poly.roots
        cold[:, i] = np.pad(found, (0, 4 - found.size), constant_values=np.nan)
    seeds += list(cold)
    roots = np.array(
        [
            np.asarray(kinetic.solve_dispersion(dielectric, k, seed, derivative=slope, continuation=False))
            for seed in seeds
        ]
    )
    growth = np.where(np.isfinite(roots), roots.imag, -np.inf)
    best = np.take_along_axis(roots, np.argmax(growth, axis=0)[None], axis=0)[0]
    return best.reshape(shape)[()]
