"""Charged-particle orbits: gyromotion, guiding-center drifts and trapped particles in a tokamak.

The formulas hold in any consistent units (SI, or Struphy's normalized units with charge and
mass in units of e and m_p) unless a function says otherwise. Vectors are arrays with a trailing
axis of length 3, (x, y, z) or any right-handed Cartesian basis; everything broadcasts.

Tokamak model
-------------
The trapped-particle functions use a large-aspect-ratio tokamak with circular flux surfaces,
B(θ) = B₀ / (1 + ε cos θ) with ε = r/R₀ the inverse aspect ratio of the flux surface and θ the
poloidal angle (θ = 0 on the outboard midplane, where B is smallest). A particle of speed v and
magnetic moment μ has the pitch parameter λ = μB₀/E = (v⊥²/v²) B₀/B, and

    v∥² = v² (1 + ε cos θ − λ) / (1 + ε cos θ) = 2ε v² (κ² − sin²(θ/2)) / (1 + ε cos θ)

with the trapping parameter κ² = (1 + ε − λ)/(2ε): particles with κ² < 1 are trapped (they bounce
at sin²(θ/2) = κ²), κ² = 0 is deeply trapped at θ = 0, κ² > 1 passes. The bounce and transit
frequencies and banana widths are to leading order in ε (the factor 1 + ε cos θ is dropped,
and the field line length is q R₀ dθ).

References
----------
P. Helander and D. J. Sigmar, Collisional Transport in Magnetized Plasmas (Cambridge, 2002),
chapter 7.

J. Wesson, Tokamaks, 4th ed. (Oxford, 2011), section 3.12.

F. F. Chen, Introduction to Plasma Physics and Controlled Fusion, 3rd ed. (Springer, 2016),
chapter 2.

Examples
--------
The bounce frequency of a barely trapped (κ² = 0.9) particle with v = 1 on the ε = 0.1 surface of
a tokamak with q = 2 and R₀ = 3:

>>> print(f"{bounce_frequency(1.0, 0.9, 0.1, 2.0, 3.0):.5f}")
0.02271
"""

from __future__ import annotations

import numpy as np

from struphy_plots.theory.special import elliptic_k


def _out(value):
    return np.asarray(value, dtype=float)[()]


def _vector(value):
    value = np.asarray(value, dtype=float)
    if value.ndim == 0 or value.shape[-1] != 3:
        raise ValueError(f"vectors need a trailing axis of length 3; got shape {value.shape}")
    return value


def _scalar(value):
    """A scalar array with an axis appended, to multiply vectors."""
    return np.asarray(value, dtype=float)[..., None]


# ---------------------------------------------------------------------------------------------
# Gyromotion
# ---------------------------------------------------------------------------------------------
def gyrofrequency(field, charge=1.0, mass=1.0):
    """Compute the gyrofrequency Ω = q|B|/m.

    Parameters
    ----------
    field : float or array_like
        Magnetic field strength |B| (its sign doesn't matter).
    charge : float or array_like, optional
        Charge q; Ω has its sign (negative for electrons). Default: ``1``.
    mass : float or array_like, optional
        Mass m. Default: ``1``.

    Returns
    -------
    float or numpy.ndarray
        Ω, in rad per unit time (a particle with q > 0 gyrates clockwise when viewed along B).

    References
    ----------
    Chen, Introduction to Plasma Physics, section 2.2.

    Examples
    --------
    >>> print(gyrofrequency(2.0, charge=-1.0, mass=0.5))
    -4.0
    """
    return _out(np.asarray(charge, dtype=float) * np.abs(np.asarray(field, dtype=float)) / mass)


def gyroradius(perpendicular_speed, field, charge=1.0, mass=1.0):
    """Compute the gyroradius ρ = m v⊥ / (|q| B).

    Parameters
    ----------
    perpendicular_speed : float or array_like
        The speed v⊥ perpendicular to B.
    field : float or array_like
        Magnetic field strength |B|.
    charge : float or array_like, optional
        Charge q (its sign doesn't matter). Default: ``1``.
    mass : float or array_like, optional
        Mass m. Default: ``1``.

    Returns
    -------
    float or numpy.ndarray
        ρ (positive).

    References
    ----------
    Chen, Introduction to Plasma Physics, section 2.2.

    Examples
    --------
    >>> print(gyroradius(3.0, 2.0))
    1.5
    """
    speed = np.abs(np.asarray(perpendicular_speed, dtype=float))
    return _out(mass * speed / np.abs(np.asarray(charge, dtype=float) * np.asarray(field, dtype=float)))


def gyroperiod(field, charge=1.0, mass=1.0):
    """Compute the gyroperiod 2π / |Ω| = 2π m / (|q| B).

    Parameters
    ----------
    field : float or array_like
        Magnetic field strength |B|.
    charge : float or array_like, optional
        Charge q (its sign doesn't matter). Default: ``1``.
    mass : float or array_like, optional
        Mass m. Default: ``1``.

    Returns
    -------
    float or numpy.ndarray
        The gyroperiod.

    References
    ----------
    Chen, Introduction to Plasma Physics, section 2.2.

    Examples
    --------
    >>> print(f"{gyroperiod(1.0):.6f}")
    6.283185
    """
    return _out(2 * np.pi / np.abs(gyrofrequency(field, charge, mass)))


def magnetic_moment(perpendicular_speed, field, mass=1.0):
    """Compute the magnetic moment μ = m v⊥² / (2B), the adiabatic invariant of the gyromotion.

    Parameters
    ----------
    perpendicular_speed : float or array_like
        The speed v⊥ perpendicular to B.
    field : float or array_like
        Magnetic field strength |B|.
    mass : float or array_like, optional
        Mass m. Default: ``1``.

    Returns
    -------
    float or numpy.ndarray
        μ.

    References
    ----------
    Chen, Introduction to Plasma Physics, section 2.3.

    Examples
    --------
    >>> print(magnetic_moment(2.0, 4.0, mass=2.0))
    1.0
    """
    speed = np.asarray(perpendicular_speed, dtype=float)
    return _out(mass * speed**2 / (2 * np.abs(np.asarray(field, dtype=float))))


# ---------------------------------------------------------------------------------------------
# Drifts
# ---------------------------------------------------------------------------------------------
def exb_drift(E, B):
    """Compute the E×B drift velocity v_E = E × B / B².

    Parameters
    ----------
    E : array_like
        Electric field, shape (..., 3).
    B : array_like
        Magnetic field, shape (..., 3).

    Returns
    -------
    numpy.ndarray
        v_E, shape (..., 3); independent of charge and mass, |v_E| = E⊥/B.

    Raises
    ------
    ValueError
        If a vector has no trailing axis of length 3.

    References
    ----------
    Chen, Introduction to Plasma Physics, section 2.2.2.

    Examples
    --------
    >>> print(exb_drift([1.0, 0.0, 0.0], [0.0, 0.0, 2.0]))
    [ 0.  -0.5  0. ]
    """
    E, B = _vector(E), _vector(B)
    return np.cross(E, B) / np.sum(B**2, axis=-1, keepdims=True) + 0.0


def grad_b_drift(perpendicular_speed, B, grad_B, charge=1.0, mass=1.0):
    """Compute the grad-B drift velocity v_∇B = (m v⊥² / (2q)) B × ∇B / B³.

    Parameters
    ----------
    perpendicular_speed : float or array_like
        The speed v⊥ perpendicular to B.
    B : array_like
        Magnetic field, shape (..., 3).
    grad_B : array_like
        Gradient of the field strength ∇|B|, shape (..., 3).
    charge : float or array_like, optional
        Charge q; ions and electrons drift in opposite directions. Default: ``1``.
    mass : float or array_like, optional
        Mass m. Default: ``1``.

    Returns
    -------
    numpy.ndarray
        v_∇B, shape (..., 3).

    Raises
    ------
    ValueError
        If a vector has no trailing axis of length 3.

    References
    ----------
    Chen, Introduction to Plasma Physics, section 2.3.1.

    Examples
    --------
    >>> print(grad_b_drift(1.0, [0.0, 0.0, 1.0], [0.1, 0.0, 0.0]))   # B along z, ∇B along x
    [0.   0.05 0.  ]
    """
    B, grad_B = _vector(B), _vector(grad_B)
    magnitude = np.linalg.norm(B, axis=-1, keepdims=True)
    energy = _scalar(mass) * _scalar(perpendicular_speed) ** 2 / 2
    return energy / _scalar(charge) * np.cross(B, grad_B) / magnitude**3 + 0.0


def curvature_drift(parallel_speed, B, curvature, charge=1.0, mass=1.0):
    """Compute the curvature drift velocity v_κ = (m v∥² / (qB)) b × κ.

    The curvature vector is κ = (b·∇)b, with b = B/|B|: it points towards the centre of
    curvature of the field line and |κ| = 1/R_c. (Chen writes the drift with the radius vector
    R_c from the centre outwards, κ = −R_c/R_c².)

    Parameters
    ----------
    parallel_speed : float or array_like
        The speed v∥ along B (its sign doesn't matter).
    B : array_like
        Magnetic field, shape (..., 3).
    curvature : array_like
        Field-line curvature κ = (b·∇)b, shape (..., 3).
    charge : float or array_like, optional
        Charge q. Default: ``1``.
    mass : float or array_like, optional
        Mass m. Default: ``1``.

    Returns
    -------
    numpy.ndarray
        v_κ, shape (..., 3).

    Raises
    ------
    ValueError
        If a vector has no trailing axis of length 3.

    References
    ----------
    Chen, Introduction to Plasma Physics, section 2.3.2.

    Examples
    --------
    A field along z curving towards −x (centre of curvature at −x, radius 10):

    >>> print(curvature_drift(1.0, [0.0, 0.0, 1.0], [-0.1, 0.0, 0.0]))
    [ 0.  -0.1  0. ]
    """
    B, curvature = _vector(B), _vector(curvature)
    magnitude = np.linalg.norm(B, axis=-1, keepdims=True)
    factor = _scalar(mass) * _scalar(parallel_speed) ** 2 / (_scalar(charge) * magnitude)
    return factor * np.cross(B / magnitude, curvature) + 0.0


def polarization_drift(dE_dt, B, charge=1.0, mass=1.0):
    """Compute the polarization drift velocity v_p = (m / (q B²)) dE⊥/dt.

    Parameters
    ----------
    dE_dt : array_like
        Time derivative of the electric field, shape (..., 3); its component along B is dropped.
    B : array_like
        Magnetic field, shape (..., 3).
    charge : float or array_like, optional
        Charge q. Default: ``1``.
    mass : float or array_like, optional
        Mass m. Default: ``1``.

    Returns
    -------
    numpy.ndarray
        v_p, shape (..., 3).

    Raises
    ------
    ValueError
        If a vector has no trailing axis of length 3.

    References
    ----------
    Chen, Introduction to Plasma Physics, section 2.5.

    Examples
    --------
    >>> print(polarization_drift([1.0, 0.0, 3.0], [0.0, 0.0, 2.0]))
    [0.25 0.   0.  ]
    """
    dE_dt, B = _vector(dE_dt), _vector(B)
    b2 = np.sum(B**2, axis=-1, keepdims=True)
    perpendicular = dE_dt - np.sum(dE_dt * B, axis=-1, keepdims=True) * B / b2
    return _scalar(mass) / (_scalar(charge) * b2) * perpendicular


def vacuum_drift(perpendicular_speed, parallel_speed, B, grad_B, charge=1.0, mass=1.0):
    """Compute the combined grad-B and curvature drift in a vacuum field (∇ × B = 0).

    In vacuum the curvature is κ = ∇⊥B / B, so v_∇B + v_κ = (m / (q B³)) (v∥² + v⊥²/2) B × ∇B.

    Parameters
    ----------
    perpendicular_speed : float or array_like
        The speed v⊥ perpendicular to B.
    parallel_speed : float or array_like
        The speed v∥ along B.
    B : array_like
        Magnetic field, shape (..., 3).
    grad_B : array_like
        Gradient of the field strength ∇|B|, shape (..., 3).
    charge : float or array_like, optional
        Charge q. Default: ``1``.
    mass : float or array_like, optional
        Mass m. Default: ``1``.

    Returns
    -------
    numpy.ndarray
        The drift velocity, shape (..., 3).

    Raises
    ------
    ValueError
        If a vector has no trailing axis of length 3.

    References
    ----------
    Chen, Introduction to Plasma Physics, section 2.3.2, eq. (2.60).

    Examples
    --------
    The vertical drift in the 1/R field of a tokamak (B toroidal along y, ∇B along −x at R = 1):

    >>> print(vacuum_drift(1.0, 1.0, [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]))
    [0.  0.  1.5]
    """
    B, grad_B = _vector(B), _vector(grad_B)
    magnitude = np.linalg.norm(B, axis=-1, keepdims=True)
    energy = _scalar(parallel_speed) ** 2 + _scalar(perpendicular_speed) ** 2 / 2
    return _scalar(mass) * energy / (_scalar(charge) * magnitude**3) * np.cross(B, grad_B) + 0.0


# ---------------------------------------------------------------------------------------------
# Trapped particles in a large-aspect-ratio tokamak
# ---------------------------------------------------------------------------------------------
_S_NODES, _S_WEIGHTS = np.polynomial.legendre.leggauss(64)
_S_NODES, _S_WEIGHTS = (_S_NODES + 1) / 2, _S_WEIGHTS / 2
# s = t³ clusters the nodes at s → 0 (λ → λ_max), where the integrand varies on the scale √ε
_S_NODES, _S_WEIGHTS = _S_NODES**3, 3 * _S_NODES**2 * _S_WEIGHTS
_THETA = 2 * np.pi * (np.arange(512) + 0.5) / 512


def _trapped_fraction_exact(epsilon):
    """1 − (3/4)⟨b²⟩ ∫₀^{1/b_max} λ dλ / ⟨√(1 − λb)⟩, ⟨A⟩ = mean over θ of A (1 + ε cos θ)."""
    eps = np.asarray(epsilon, dtype=float)[..., None, None]
    weight = 1 + eps * np.cos(_THETA)  # ∝ R ∝ 1/B, for concentric circles
    b = 1 / weight
    lam_max = 1 - eps
    lam = lam_max * (1 - _S_NODES[:, None] ** 2)  # λ = λ_max (1 − s²) removes the endpoint singularity
    root = np.mean(np.sqrt(np.clip(1 - lam * b, 0, None)) * weight, axis=-1)  # ⟨√(1 − λb)⟩
    integrand = lam[..., 0] * 2 * lam_max[..., 0] * _S_NODES / root
    b2 = np.mean(b**2 * weight, axis=-1)[..., 0]
    return 1 - 0.75 * b2 * np.sum(_S_WEIGHTS * integrand, axis=-1)


def trapped_fraction(epsilon, approximation="exact"):
    """Compute the effective fraction of trapped particles on a flux surface of a circular tokamak.

    f_t = 1 − (3/4) ⟨B²⟩ ∫₀^{1/B_max} λ dλ / ⟨√(1 − λB)⟩ with B = B₀/(1 + ε cos θ) and the
    flux-surface average ⟨A⟩ = ∮ A (1 + ε cos θ) dθ / 2π, the fraction that enters neoclassical
    transport (bootstrap current, neoclassical resistivity).

    Parameters
    ----------
    epsilon : float or array_like
        Inverse aspect ratio ε = r/R₀ of the flux surface, 0 ≤ ε < 1.
    approximation : {"exact", "lin-liu", "sqrt"}, optional
        ``"exact"``: the integral above, by quadrature (relative accuracy 1e-8 or better for ε ≥ 1e-8);
        ``"lin-liu"``: Lin-Liu and Miller's fit 1 − (1 − ε)² / (√(1 − ε²) (1 + 1.46 √ε)), within
        a few per cent at any ε; ``"sqrt"``: the small-ε limit 1.46 √ε. Default: ``"exact"``.

    Returns
    -------
    float or numpy.ndarray
        f_t, 0 at ε = 0 and 1 at ε = 1.

    Raises
    ------
    ValueError
        If ``approximation`` is unknown.

    References
    ----------
    Helander and Sigmar, Collisional Transport in Magnetized Plasmas, section 11.2 (f_t ≈ 1.46 √ε).

    Y. R. Lin-Liu and R. L. Miller, "Upper and lower bounds of the effective trapped particle
    fraction in general tokamak equilibria", Phys. Plasmas 2, 1666 (1995).

    Examples
    --------
    >>> print(f"{trapped_fraction(0.1):.4f}, {trapped_fraction(0.1, 'lin-liu'):.4f}, "
    ...       f"{trapped_fraction(0.1, 'sqrt'):.4f}")
    0.4492, 0.4431, 0.4617
    """
    eps = np.asarray(epsilon, dtype=float)
    if approximation == "sqrt":
        return _out(1.46 * np.sqrt(eps))
    if approximation == "lin-liu":
        with np.errstate(divide="ignore", invalid="ignore"):
            value = 1 - (1 - eps) ** 2 / (np.sqrt(1 - eps**2) * (1 + 1.46 * np.sqrt(eps)))
        return _out(np.where(eps == 1, 1.0, value))
    if approximation == "exact":
        with np.errstate(divide="ignore", invalid="ignore"):
            value = _trapped_fraction_exact(np.where(eps == 1, 0.5, eps))
        return _out(np.where(eps == 1, 1.0, np.where(eps == 0, 0.0, value)))
    raise ValueError(f"approximation must be 'exact', 'lin-liu' or 'sqrt'; got {approximation!r}")


def pitch_parameter(pitch, epsilon, theta=0.0):
    """Compute the pitch parameter λ = μB₀/E from the pitch v∥/v at the poloidal angle θ.

    λ = (1 − (v∥/v)²) B₀/B(θ) = (1 − (v∥/v)²)(1 + ε cos θ).

    Parameters
    ----------
    pitch : float or array_like
        v∥/v, between −1 and 1, at the angle ``theta``.
    epsilon : float or array_like
        Inverse aspect ratio ε of the flux surface.
    theta : float or array_like, optional
        Poloidal angle θ where the pitch is given (0: outboard midplane). Default: ``0``.

    Returns
    -------
    float or numpy.ndarray
        λ.

    References
    ----------
    Helander and Sigmar, Collisional Transport in Magnetized Plasmas, section 7.1.

    Examples
    --------
    >>> print(f"{pitch_parameter(0.3, 0.1):.4f}")
    1.0010
    """
    pitch = np.asarray(pitch, dtype=float)
    eps = np.asarray(epsilon, dtype=float)
    return _out((1 - pitch**2) * (1 + eps * np.cos(theta)))


def trapping_boundary(epsilon, quantity="lambda", theta=0.0):
    """Compute the trapped–passing boundary on a flux surface.

    Particles with λ = μB₀/E > λ_c = B₀/B_max = 1 − ε are trapped, equivalently those with a
    pitch |v∥/v| < √(1 − (1 − ε)/(1 + ε cos θ)) at the angle θ (√(2ε/(1 + ε)) on the outboard
    midplane).

    Parameters
    ----------
    epsilon : float or array_like
        Inverse aspect ratio ε of the flux surface.
    quantity : {"lambda", "pitch"}, optional
        ``"lambda"``: the critical λ_c; ``"pitch"``: the critical |v∥/v| at ``theta``.
        Default: ``"lambda"``.
    theta : float or array_like, optional
        Poloidal angle for ``quantity="pitch"``. Default: ``0``.

    Returns
    -------
    float or numpy.ndarray
        λ_c or the critical pitch.

    Raises
    ------
    ValueError
        If ``quantity`` is unknown.

    References
    ----------
    Wesson, Tokamaks, section 3.12.

    Examples
    --------
    >>> print(f"{trapping_boundary(0.1):.2f}, {trapping_boundary(0.1, 'pitch'):.4f}")
    0.90, 0.4264
    """
    eps = np.asarray(epsilon, dtype=float)
    lam_c = 1 - eps
    if quantity == "lambda":
        return _out(lam_c)
    if quantity == "pitch":
        return _out(np.sqrt(1 - lam_c / (1 + eps * np.cos(theta))))
    raise ValueError(f"quantity must be 'lambda' or 'pitch'; got {quantity!r}")


def trapping_parameter(lam, epsilon):
    """Compute the trapping parameter κ² = (1 + ε − λ) / (2ε).

    κ² < 1 for trapped particles, which bounce at sin²(θ/2) = κ²; κ² = 0 for particles deeply
    trapped at θ = 0 (λ = 1 + ε), κ² = 1 on the trapped–passing boundary (λ = 1 − ε) and κ² > 1
    for passing ones.

    Parameters
    ----------
    lam : float or array_like
        Pitch parameter λ = μB₀/E (see :func:`pitch_parameter`).
    epsilon : float or array_like
        Inverse aspect ratio ε > 0 of the flux surface.

    Returns
    -------
    float or numpy.ndarray
        κ².

    References
    ----------
    Helander and Sigmar, Collisional Transport in Magnetized Plasmas, section 7.2.

    Examples
    --------
    >>> print(f"{trapping_parameter(1.0, 0.1):.2f}")
    0.50
    """
    lam = np.asarray(lam, dtype=float)
    eps = np.asarray(epsilon, dtype=float)
    return _out((1 + eps - lam) / (2 * eps))


def bounce_frequency(speed, kappa2, epsilon, safety_factor, major_radius):
    """Compute the bounce frequency of a trapped particle, ω_b = π v √(2ε) / (4 q R₀ K(κ²)).

    The period of the banana orbit is τ_b = 2π/ω_b = 8 q R₀ K(κ²) / (v √(2ε)), to leading order
    in ε. Deeply trapped particles (κ² = 0) bounce at ω_b = v √(ε/2) / (q R₀); ω_b → 0
    logarithmically at the trapped–passing boundary κ² → 1.

    Parameters
    ----------
    speed : float or array_like
        The particle speed v.
    kappa2 : float or array_like
        Trapping parameter κ² (see :func:`trapping_parameter`), 0 ≤ κ² < 1.
    epsilon : float or array_like
        Inverse aspect ratio ε of the flux surface.
    safety_factor : float or array_like
        Safety factor q of the flux surface.
    major_radius : float or array_like
        Major radius R₀.

    Returns
    -------
    float or numpy.ndarray
        ω_b in rad per unit time; ``nan`` for passing particles (κ² > 1).

    References
    ----------
    Helander and Sigmar, Collisional Transport in Magnetized Plasmas, section 7.2.

    Wesson, Tokamaks, section 3.12 (the deeply trapped limit).

    Examples
    --------
    >>> print(f"{bounce_frequency(1.0, 0.0, 0.02, 1.0, 1.0):.4f}")   # √(ε/2) deeply trapped
    0.1000
    """
    k2 = np.asarray(kappa2, dtype=float)
    speed = np.abs(np.asarray(speed, dtype=float))
    eps = np.asarray(epsilon, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = elliptic_k(np.where((k2 >= 0) & (k2 <= 1), k2, np.nan))
        omega = np.pi * speed * np.sqrt(2 * eps) / (4 * np.abs(safety_factor) * major_radius * k)
    return _out(omega)


def transit_frequency(speed, kappa2, epsilon, safety_factor, major_radius):
    """Compute the poloidal transit frequency of a passing particle.

    ω_t = π κ v √(2ε) / (2 q R₀ K(1/κ²)), to leading order in ε: 2π/ω_t is the time for one
    poloidal turn. Far from the boundary (κ² ≫ 1) it tends to |v∥| / (q R₀); ω_t → 0
    logarithmically at κ² → 1.

    Parameters
    ----------
    speed : float or array_like
        The particle speed v.
    kappa2 : float or array_like
        Trapping parameter κ² (see :func:`trapping_parameter`), κ² > 1.
    epsilon : float or array_like
        Inverse aspect ratio ε of the flux surface.
    safety_factor : float or array_like
        Safety factor q of the flux surface.
    major_radius : float or array_like
        Major radius R₀.

    Returns
    -------
    float or numpy.ndarray
        ω_t in rad per unit time; ``nan`` for trapped particles (κ² < 1).

    References
    ----------
    Helander and Sigmar, Collisional Transport in Magnetized Plasmas, section 7.2.

    Examples
    --------
    >>> print(f"{transit_frequency(1.0, 50.0, 0.01, 1.0, 1.0):.4f}")   # ≈ v∥/(qR₀) = 1
    0.9950
    """
    k2 = np.asarray(kappa2, dtype=float)
    speed = np.abs(np.asarray(speed, dtype=float))
    eps = np.asarray(epsilon, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        k2 = np.where(k2 >= 1, k2, np.nan)
        k = elliptic_k(1 / k2)
        omega = np.pi * np.sqrt(k2) * speed * np.sqrt(2 * eps) / (2 * np.abs(safety_factor) * major_radius * k)
    return _out(omega)


def banana_width(gyroradius, kappa2, epsilon, safety_factor):
    """Compute the full radial width of a banana orbit, Δr = 2√2 q κ ρ / √ε.

    From the conservation of the canonical toroidal momentum, the radial excursion is
    δr = q δv∥ / (ε Ω); v∥ swings between ±κ v √(2ε) at the outboard midplane. The width is
    largest near the trapped–passing boundary, 2√2 q ρ / √ε, the "q ρ / √ε" of scaling estimates,
    and vanishes for deeply trapped particles.

    Parameters
    ----------
    gyroradius : float or array_like
        Gyroradius ρ = m v / (|q| B₀) with the total speed v (for trapped particles v⊥ ≈ v).
    kappa2 : float or array_like
        Trapping parameter κ², 0 ≤ κ² ≤ 1.
    epsilon : float or array_like
        Inverse aspect ratio ε of the flux surface.
    safety_factor : float or array_like
        Safety factor q of the flux surface.

    Returns
    -------
    float or numpy.ndarray
        Δr, in the units of ``gyroradius``; ``nan`` for passing particles (κ² > 1).

    References
    ----------
    Helander and Sigmar, Collisional Transport in Magnetized Plasmas, section 7.3.

    Wesson, Tokamaks, section 3.12.

    Examples
    --------
    >>> print(f"{banana_width(0.01, 1.0, 0.1, 2.0):.4f}")
    0.1789
    """
    k2 = np.asarray(kappa2, dtype=float)
    rho = np.abs(np.asarray(gyroradius, dtype=float))
    eps = np.asarray(epsilon, dtype=float)
    with np.errstate(invalid="ignore"):
        kappa = np.sqrt(np.where((k2 >= 0) & (k2 <= 1), k2, np.nan))
        return _out(2 * np.sqrt(2) * np.abs(safety_factor) * kappa * rho / np.sqrt(eps))
