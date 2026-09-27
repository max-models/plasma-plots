"""Kinetic dispersion relations of unmagnetized plasmas: Langmuir waves and Landau damping,
ion-acoustic waves, beam-plasma, two-stream and bump-on-tail instabilities, and the Weibel
instability.

Units
-----
Electrostatic results are normalized to the reference electrons: time in 1/ω_pe, length in the
Debye length λ_De and velocity in the thermal speed v_the = √(T_e/m_e), so that
ω_pe = v_the/λ_De = 1. Densities are relative to the reference electron density, charges in e and
masses in m_e. A Maxwellian species of thermal speed v_th = √(T/m) drifting at u enters through
ζ = (ω − k u)/(√2 |k| v_th). The electromagnetic Weibel relation (:func:`weibel`) uses c
instead: k in ω_pe/c and speeds in c.

All frequencies are complex for ``exp(i(kx − ωt))``: a positive imaginary part is a growth rate,
a negative one a damping rate. Functions take scalars or arrays that broadcast against each
other, and return a Python complex for scalar input, else a complex numpy array.

Examples
--------
>>> omega = langmuir(0.5)
>>> round(omega.real, 6), round(omega.imag, 6)
(1.415662, -0.153359)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from struphy_plots.theory.special import faddeeva, plasma_dispersion

#: Proton-to-electron mass ratio (CODATA 2018), the default ion mass in units of m_e.
PROTON_ELECTRON_MASS_RATIO = 1836.15267343

_SQRT2 = np.sqrt(2.0)
_SQRT_PI = np.sqrt(np.pi)

# 1 + ζ Z(ζ) for |ζ| ≥ _FAR by its asymptotic series −Σ cₙ ζ⁻²ⁿ, cₙ = (2n − 1)!!/2ⁿ, which avoids
# the cancellation of 1 and ζ Z(ζ) ≈ −1 at large |ζ| (the cold limit). 36 terms (the optimal
# truncation at |ζ| = 6) leave a relative error of ~2e-14 there, much less further out.
_FAR = 6.0
_SERIES = np.cumprod(np.arange(1, 72, 2) / 2.0)


def _out(value):
    """A Python complex for 0-d input, else the complex array."""
    value = np.asarray(value, dtype=complex)
    return complex(value) if value.ndim == 0 else value


def _floats(*values):
    """The values as float arrays broadcast against each other."""
    return np.broadcast_arrays(*(np.asarray(value, dtype=float) for value in values))


def _real(value):
    """A Python float for 0-d input, else the array."""
    return float(value) if np.ndim(value) == 0 else value


def _response(zeta):
    """W(ζ) = 1 + ζ Z(ζ) and its derivative W'(ζ) = Z − 2ζW, accurate also at large |ζ|."""
    zeta = np.asarray(zeta, dtype=complex)
    far = np.abs(zeta) >= _FAR
    near_zeta = np.where(far, 0.5, zeta)
    z = plasma_dispersion(near_zeta)
    w_near = 1 + near_zeta * z
    dw_near = z - 2 * near_zeta * w_near
    far_zeta = np.where(far, zeta, _FAR)
    u = 1 / far_zeta**2
    n = np.arange(1, _SERIES.size + 1)
    w_far = -u * np.polyval(_SERIES[::-1], u)
    dw_far = 2 / far_zeta * u * np.polyval((n * _SERIES)[::-1], u)
    # The Landau (residue) term i√π σ ζ exp(−ζ²), with the Stokes multiplier σ switching from 0
    # (upper half plane) through 1 (real axis) to 2 (lower half plane), smoothed across the real
    # axis after Berry: σ = erfc(√2 |x| y/√(x² − y²)) for |x| > |y|, ζ = x + iy.
    x, y = np.abs(far_zeta.real), far_zeta.imag
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        t = np.where(x > np.abs(y), _SQRT2 * x * y / np.sqrt(np.maximum(x**2 - y**2, 0.0)), np.sign(y) * np.inf)
        t = np.clip(t, -30.0, 30.0)
        tail = (np.exp(-(t**2)) * faddeeva(1j * np.abs(t))).real  # erfc(|t|)
        sigma = np.where(t >= 0, tail, 2 - tail)
        gauss = np.where(sigma > 0, sigma * np.exp(-(far_zeta**2)), 0)
        w_far = w_far + 1j * _SQRT_PI * far_zeta * gauss
        dw_far = dw_far + 1j * _SQRT_PI * (1 - 2 * far_zeta**2) * gauss
    return np.where(far, w_far, w_near), np.where(far, dw_far, dw_near)


@dataclass(frozen=True)
class Maxwellian:
    """A drifting Maxwellian species, in the normalized units of this module.

    f(v) = n/(√(2π) v_th) exp(−(v − u)²/(2 v_th²)). The fields may also be arrays that broadcast
    against the wavenumbers.

    Attributes
    ----------
    density : float
        The density n_s, relative to the reference electron density. Default: ``1.0``.
    charge : float
        The charge q_s in units of e. Default: ``-1.0`` (electrons).
    mass : float
        The mass m_s in units of m_e. Default: ``1.0``.
    thermal_speed : float
        The thermal speed v_th,s = √(T_s/m_s) in units of v_the, positive. Default: ``1.0``.
    drift : float
        The drift speed u_s in units of v_the. Default: ``0.0``.

    Examples
    --------
    >>> Maxwellian().plasma_frequency
    1.0
    >>> beam = Maxwellian(density=0.1, thermal_speed=0.5, drift=4.5)
    >>> round(float(beam.plasma_frequency), 4)
    0.3162
    """

    density: float = 1.0
    charge: float = -1.0
    mass: float = 1.0
    thermal_speed: float = 1.0
    drift: float = 0.0

    @property
    def plasma_frequency(self):
        """The plasma frequency ω_ps = √(n_s q_s²/m_s), in units of ω_pe."""
        return _real(np.sqrt(np.asarray(self.density * self.charge**2 / self.mass, dtype=float)))

    @classmethod
    def ions(cls, mass_ratio=PROTON_ELECTRON_MASS_RATIO, temperature_ratio=1.0, charge=1.0, drift=0.0):
        """Create quasi-neutral ions for the reference electrons.

        Parameters
        ----------
        mass_ratio : float, optional
            m_i/m_e. Default: the proton mass ratio, 1836.15.
        temperature_ratio : float, optional
            T_e/T_i. Default: ``1.0``.
        charge : float, optional
            The charge number Z; the density is 1/Z. Default: ``1.0``.
        drift : float, optional
            The drift speed in units of v_the. Default: ``0.0``.

        Returns
        -------
        Maxwellian
            Ions with thermal speed √(T_i/m_i) = 1/√(mass_ratio · temperature_ratio) in units of
            v_the.

        Examples
        --------
        >>> ions = Maxwellian.ions(mass_ratio=100.0, temperature_ratio=4.0)
        >>> ions.thermal_speed, round(float(ions.plasma_frequency), 4)
        (0.05, 0.1)
        """
        return cls(
            density=1.0 / charge,
            charge=charge,
            mass=mass_ratio,
            thermal_speed=_real(1.0 / np.sqrt(np.asarray(mass_ratio * temperature_ratio, dtype=float))),
            drift=drift,
        )


_ELECTRONS = (Maxwellian(),)


def susceptibility(omega, k, species, derivative=0):
    """Compute the electrostatic susceptibility χ_s(ω, k) of one Maxwellian species.

    χ_s = (ω_ps²/(k² v_th,s²)) [1 + ζ_s Z(ζ_s)], with ζ_s = (ω − k u_s)/(√2 |k| v_th,s) and Z the
    plasma dispersion function (Landau's continuation, valid for any complex ω). For a negative k
    the drift enters as k u_s, so χ_s(ω, −k) is χ_s(ω, k) with the drift reversed.

    Parameters
    ----------
    omega : complex or array_like
        The complex frequency in units of ω_pe.
    k : float or array_like
        The wavenumber in units of 1/λ_De, nonzero.
    species : Maxwellian
        The species.
    derivative : {0, 1}, optional
        0 for χ_s, 1 for ∂χ_s/∂ω. Default: ``0``.

    Returns
    -------
    complex or numpy.ndarray
        χ_s or ∂χ_s/∂ω, broadcast over ``omega``, ``k`` and the species fields.

    Raises
    ------
    ValueError
        If ``derivative`` is not 0 or 1.

    References
    ----------
    B. D. Fried and S. D. Conte, The Plasma Dispersion Function (Academic Press, 1961).
    T. H. Stix, Waves in Plasmas (AIP, 1992), ch. 8.

    Examples
    --------
    The cold limit χ → −ω_p²/ω²:

    >>> round(susceptibility(10.0, 0.01, Maxwellian()).real, 6)
    -0.01
    """
    if derivative not in (0, 1):
        raise ValueError(f"derivative must be 0 or 1; got {derivative!r}")
    omega = np.asarray(omega, dtype=complex)
    k = np.asarray(k, dtype=float)
    thermal_speed = np.asarray(species.thermal_speed, dtype=float)
    scale = _SQRT2 * np.abs(k) * thermal_speed
    with np.errstate(divide="ignore", invalid="ignore"):
        zeta = (omega - k * species.drift) / scale
        w, dw = _response(zeta)
        prefactor = species.plasma_frequency**2 / (k * thermal_speed) ** 2
        return _out(prefactor * w if derivative == 0 else prefactor * dw / scale)


def electrostatic_dielectric(omega, k, species=None, derivative=0):
    """Compute the electrostatic dielectric function ε(ω, k) = 1 + Σ_s χ_s of Maxwellian species.

    Its zeros are the electrostatic (Langmuir, ion-acoustic, beam) modes.

    Parameters
    ----------
    omega : complex or array_like
        The complex frequency in units of ω_pe.
    k : float or array_like
        The wavenumber in units of 1/λ_De, nonzero.
    species : Maxwellian or sequence of Maxwellian, optional
        The mobile species. Default: the reference electrons, ``Maxwellian()``, with immobile
        ions.
    derivative : {0, 1}, optional
        0 for ε, 1 for ∂ε/∂ω. Default: ``0``.

    Returns
    -------
    complex or numpy.ndarray
        ε or ∂ε/∂ω, broadcast over ``omega``, ``k`` and the species fields.

    See Also
    --------
    susceptibility : One species' contribution χ_s.

    Examples
    --------
    >>> abs(electrostatic_dielectric(langmuir(0.3), 0.3)) < 1e-10
    True
    """
    if species is None:
        species = _ELECTRONS
    elif isinstance(species, Maxwellian):
        species = (species,)
    total = 0 if derivative else 1
    for s in species:
        total = total + np.asarray(susceptibility(omega, k, s, derivative=derivative))
    return _out(total)


# ---------------------------------------------------------------------------------------------
# Root finding
# ---------------------------------------------------------------------------------------------
def _finite_difference(function):
    """A central-difference derivative in ω of an analytic ``function(omega)``."""

    def derivative(omega):
        h = 1e-6 * (1 + np.abs(omega))
        return (function(omega + h) - function(omega - h)) / (2 * h)

    return derivative


def _newton(function, derivative, omega, tol=1e-11, maxiter=60):
    """Vectorized damped Newton iteration; returns the roots and which of them converged.

    A step that does not decrease |f| is halved (up to 12 times). An element has converged once
    its full Newton step is below ``tol · |ω| + 1e-14`` (that step is still taken).
    """
    omega = np.array(omega, dtype=complex)
    with np.errstate(all="ignore"):
        f = np.asarray(function(omega), dtype=complex)
        omega, f = np.broadcast_arrays(omega, f)
        omega, f = omega.copy(), f.copy()
        active = np.isfinite(omega) & np.isfinite(f)
        converged = np.zeros(omega.shape, dtype=bool)
        for _ in range(maxiter):
            if not active.any():
                break
            step = np.where(active, f / np.asarray(derivative(omega)), 0)
            active &= np.isfinite(step)
            step = np.where(active, step, 0)
            lam = np.ones(omega.shape)
            trial = omega - step
            f_trial = np.asarray(function(trial), dtype=complex)
            for _ in range(12):
                worse = active & ~(np.abs(f_trial) <= np.abs(f))
                if not worse.any():
                    break
                lam = np.where(worse, lam / 2, lam)
                trial = np.where(worse, omega - lam * step, trial)
                f_trial = np.where(worse, function(trial), f_trial)
            done = active & (np.abs(step) <= tol * np.abs(omega) + 1e-14)
            omega = np.where(active, trial, omega)
            f = np.where(active, f_trial, f)
            converged |= done & np.isfinite(omega)
            active &= ~done & np.isfinite(omega) & np.isfinite(f)
    return omega, converged


def _track(function, derivative, guess, steps):
    """Follow a root of ``function(omega, s)`` from s = 0 (seeded by ``guess``) to s = 1.

    Each step is predicted by linear extrapolation of the last two roots and corrected by Newton
    (retried from the last root if that fails). Elements that fail become ``nan``.
    """
    s_values = np.linspace(0.0, 1.0, steps + 1)
    omega, ok = _newton(lambda w: function(w, 0.0), lambda w: derivative(w, 0.0), guess)
    previous = omega
    for i, s in enumerate(s_values[1:], start=1):
        predicted = omega if i == 1 else 2 * omega - previous
        new, converged = _newton(lambda w: function(w, s), lambda w: derivative(w, s), predicted)
        if not converged.all():
            retry, retried = _newton(lambda w: function(w, s), lambda w: derivative(w, s), omega)
            new = np.where(converged, new, retry)
            converged |= retried
        ok &= converged
        previous, omega = omega, np.where(ok, new, omega)
    return np.where(ok, omega, np.nan + 1j * np.nan)


def _path_steps(*ratios):
    """The number of continuation steps for log-spaced paths spanning these ratios."""
    span = max(float(np.max(np.abs(np.log(np.asarray(r, dtype=float))), initial=0.0)) for r in ratios)
    return int(max(12, np.ceil(20 * span)))


def solve_dispersion(function, k, guess, derivative=None, continuation=True, tol=1e-11, maxiter=60):
    """Find a complex root ω(k) of a dispersion relation for every wavenumber.

    Damped Newton iteration in ω, with the analytic derivative when one is given and a
    central difference otherwise. With ``continuation``, the wavenumbers are solved in the given
    order and each root seeds the next (linearly extrapolated from the last two), which follows one
    branch along a k-grid; otherwise every k starts from ``guess`` and all are solved at once.

    Parameters
    ----------
    function : callable
        ``function(omega, k)``, analytic in ω, whose zero is sought, e.g.
        ``lambda w, k: electrostatic_dielectric(w, k, species)``. It must broadcast over arrays.
    k : float or array_like
        The wavenumbers; one-dimensional with ``continuation``, else any shape.
    guess : complex or array_like
        The starting frequency: a scalar for the first k with ``continuation``, else anything that
        broadcasts against ``k``.
    derivative : callable, optional
        ``derivative(omega, k)``, the ω-derivative of ``function``. Default: a central difference.
    continuation : bool, optional
        Continue the root along ``k``. Default: ``True``.
    tol : float, optional
        Converged once the Newton step is below ``tol · |ω|``. Default: ``1e-11``.
    maxiter : int, optional
        The maximum number of Newton steps per wavenumber. Default: ``60``.

    Returns
    -------
    complex or numpy.ndarray
        ω for every k, ``nan`` where Newton did not converge (no exception is raised); with
        ``continuation`` the next wavenumber is then seeded by the last converged root.

    Raises
    ------
    ValueError
        With ``continuation``, if ``k`` has more than one dimension or ``guess`` is not a scalar.

    Examples
    --------
    The Langmuir branch followed from k = 0.2 to 0.5:

    >>> k = np.linspace(0.2, 0.5, 7)
    >>> omega = solve_dispersion(electrostatic_dielectric, k, guess=1.06)
    >>> round(complex(omega[-1]).real, 4), round(complex(omega[-1]).imag, 4)
    (1.4157, -0.1534)
    """
    if derivative is None:

        def slope(omega, kk):
            return _finite_difference(lambda w: function(w, kk))(omega)
    else:
        slope = derivative
    k = np.asarray(k, dtype=float)
    if not continuation:
        omega, converged = _newton(
            lambda w: function(w, k), lambda w: slope(w, k), np.broadcast_to(guess, k.shape), tol, maxiter
        )
        return _out(np.where(converged, omega, np.nan + 1j * np.nan))
    if k.ndim > 1:
        raise ValueError(f"continuation needs a one-dimensional k; got shape {k.shape}")
    if np.ndim(guess) != 0:
        raise ValueError("continuation needs a scalar guess (the root at the first k)")
    out = np.full(np.atleast_1d(k).shape, np.nan + 1j * np.nan)
    good = []  # the last two converged roots
    for i, kk in enumerate(np.atleast_1d(k)):
        seed = complex(guess) if not good else (good[-1] if len(good) == 1 else 2 * good[-1] - good[-2])
        omega, converged = _newton(lambda w: function(w, kk), lambda w: slope(w, kk), seed, tol, maxiter)
        if not converged and good:
            omega, converged = _newton(lambda w: function(w, kk), lambda w: slope(w, kk), good[-1], tol, maxiter)
        if converged:
            out[i] = complex(omega)
            good = (good + [complex(omega)])[-2:]
    return _out(out.reshape(k.shape))


def _most_unstable(function, derivative, guesses):
    """The root with the largest imaginary part found from several guesses (last axis)."""
    omega, converged = _newton(function, derivative, guesses)
    omega = np.where(converged, omega, np.nan + 1j * np.nan)
    growth = np.where(converged, omega.imag, -np.inf)
    best = np.argmax(growth, axis=-1)
    return np.take_along_axis(omega, best[..., None], axis=-1)[..., 0]


def _bisect(function, a, b, iterations=200):
    """Bisect a real ``function(x)`` between ``a`` and ``b`` (opposite signs, any order), elementwise."""
    fa = function(a)
    for _ in range(iterations):
        mid = (a + b) / 2
        fm = function(mid)
        same = np.sign(fm) == np.sign(fa)
        a, fa, b = np.where(same, mid, a), np.where(same, fm, fa), np.where(same, b, mid)
        if np.all(np.abs(b - a) <= 4e-16 * np.maximum(np.abs(a), np.abs(b)) + 1e-300):
            break
    return (a + b) / 2


# ---------------------------------------------------------------------------------------------
# Langmuir waves
# ---------------------------------------------------------------------------------------------
def bohm_gross(k):
    """Compute the Bohm–Gross frequency ω = √(1 + 3k²) of Langmuir waves.

    The fluid (adiabatic, γ = 3) Langmuir frequency, the real part of :func:`langmuir` for
    k λ_De ≪ 1.

    Parameters
    ----------
    k : float or array_like
        The wavenumber in units of 1/λ_De.

    Returns
    -------
    complex or numpy.ndarray
        ω in units of ω_pe (real-valued).

    References
    ----------
    D. Bohm and E. P. Gross, "Theory of plasma oscillations. A. Origin of medium-like
    behavior", Phys. Rev. 75, 1851 (1949).

    Examples
    --------
    >>> bohm_gross(0.0)
    (1+0j)
    >>> round(bohm_gross(0.5).real, 4)
    1.3229
    """
    return _out(np.sqrt(1 + 3 * np.asarray(k, dtype=float) ** 2))


def landau_damping_weak(k):
    """Compute the weak-damping approximation of Langmuir waves: Bohm–Gross plus Landau damping.

    ω = √(1 + 3k²) − i √(π/8) k⁻³ exp(−1/(2k²) − 3/2), the textbook expansion for k λ_De ≪ 1
    (Chen, eq. 7.133). It overestimates the damping at moderate k (by ~60 % at k = 0.3); use
    :func:`langmuir` for the exact root.

    Parameters
    ----------
    k : float or array_like
        The wavenumber in units of 1/λ_De, nonzero.

    Returns
    -------
    complex or numpy.ndarray
        ω in units of ω_pe.

    References
    ----------
    F. F. Chen, Introduction to Plasma Physics and Controlled Fusion, 3rd ed. (Springer, 2016),
    sec. 7.5.

    Examples
    --------
    >>> omega = landau_damping_weak(0.2)
    >>> round(omega.real, 4), round(omega.imag, 7)
    (1.0583, -6.51e-05)
    """
    k = np.abs(np.asarray(k, dtype=float))
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        gamma = np.sqrt(np.pi / 8) / k**3 * np.exp(-1 / (2 * k**2) - 1.5)
    gamma = np.where(k == 0, 0.0, gamma)
    return _out(np.sqrt(1 + 3 * k**2) - 1j * gamma)


def langmuir(k):
    """Compute the exact kinetic Langmuir root: the least-damped zero of ε(ω, k) near Bohm–Gross.

    Electrons (Maxwellian, v_th = 1) with immobile ions. The root is continued in k from
    k λ_De = 0.1, where the weak-damping approximation is accurate.

    Parameters
    ----------
    k : float or array_like
        The wavenumber in units of 1/λ_De.

    Returns
    -------
    complex or numpy.ndarray
        ω in units of ω_pe, with ω(−k) = ω(k) and ω(0) = 1; ``nan`` where the root was lost.

    See Also
    --------
    landau_damping_weak : The small-k approximation.

    References
    ----------
    L. D. Landau, "On the vibrations of the electronic plasma", J. Phys. USSR 10, 25 (1946).
    J. Canosa, "Numerical solution of Landau's dispersion equation", J. Comput. Phys. 13, 158
    (1973).

    Examples
    --------
    >>> omega = langmuir([0.3, 0.4, 0.5, 1.0])
    >>> np.round(omega, 4)
    array([1.1598-0.0126j, 1.2851-0.0661j, 1.4157-0.1534j, 2.0459-0.8513j])
    """
    k = np.abs(np.asarray(k, dtype=float))
    start = 0.1
    target = np.where(k == 0, start, k)

    def wavenumber(s):
        return start * (target / start) ** s

    def function(omega, s):
        return electrostatic_dielectric(omega, wavenumber(s))

    def derivative(omega, s):
        return electrostatic_dielectric(omega, wavenumber(s), derivative=1)

    guess = np.full(k.shape, landau_damping_weak(start))
    omega = _track(function, derivative, guess, _path_steps(target / start))
    return _out(np.where(k == 0, 1.0, omega))


# ---------------------------------------------------------------------------------------------
# Ion-acoustic waves
# ---------------------------------------------------------------------------------------------
def ion_acoustic_fluid(k, temperature_ratio=10.0, mass_ratio=PROTON_ELECTRON_MASS_RATIO, adiabatic_index=3.0):
    """Compute the fluid ion-acoustic frequency with Boltzmann electrons and adiabatic ions.

    ω² = k² (T_e/(1 + k² λ_De²) + γ_i T_i)/m_i, i.e. in normalized units
    ω = k √((1/(1 + k²) + γ_i/τ)/μ) with τ = T_e/T_i and μ = m_i/m_e; for k λ_De ≪ 1,
    ω = k c_s with c_s = √((T_e + γ_i T_i)/m_i).

    Parameters
    ----------
    k : float or array_like
        The wavenumber in units of 1/λ_De.
    temperature_ratio : float or array_like, optional
        τ = T_e/T_i. Default: ``10.0``.
    mass_ratio : float or array_like, optional
        μ = m_i/m_e. Default: the proton mass ratio, 1836.15.
    adiabatic_index : float, optional
        γ_i of the ions. Default: ``3.0`` (one-dimensional adiabatic compression).

    Returns
    -------
    complex or numpy.ndarray
        ω in units of ω_pe (real-valued).

    References
    ----------
    F. F. Chen, Introduction to Plasma Physics and Controlled Fusion, 3rd ed. (Springer, 2016),
    sec. 4.6.

    Examples
    --------
    >>> round(ion_acoustic_fluid(0.1, temperature_ratio=10.0, mass_ratio=100.0).real, 5)
    0.01136
    """
    k = np.asarray(k, dtype=float)
    return _out(k * np.sqrt((1 / (1 + k**2) + adiabatic_index / np.asarray(temperature_ratio)) / mass_ratio))


def ion_acoustic(k, temperature_ratio=10.0, mass_ratio=PROTON_ELECTRON_MASS_RATIO):
    """Compute the kinetic ion-acoustic root of Maxwellian electrons and ions.

    The zero of ε = 1 + χ_e + χ_i with the reference electrons and ions of
    :meth:`Maxwellian.ions`, damped by electron and ion Landau damping. It is continued from
    k λ_De = 0.01 and T_e/T_i = max(τ, 30), seeded with the fluid frequency and the weak damping
    γ/ω = −√(π/8) (√(m_e/m_i) + τ^(3/2) exp(−τ/2 − 3/2)) (Chen, eq. 7.144).

    Parameters
    ----------
    k : float or array_like
        The wavenumber in units of 1/λ_De, positive (ω(−k) = −conj ω(k)).
    temperature_ratio : float or array_like, optional
        τ = T_e/T_i. Default: ``10.0``.
    mass_ratio : float or array_like, optional
        μ = m_i/m_e. Default: the proton mass ratio, 1836.15.

    Returns
    -------
    complex or numpy.ndarray
        ω in units of ω_pe; ``nan`` where the root was lost.

    See Also
    --------
    ion_acoustic_fluid : The fluid estimate.

    References
    ----------
    F. F. Chen, Introduction to Plasma Physics and Controlled Fusion, 3rd ed. (Springer, 2016),
    sec. 7.6.
    B. D. Fried and R. W. Gould, "Longitudinal ion oscillations in a hot plasma", Phys. Fluids 4,
    139 (1961).

    Examples
    --------
    >>> omega = ion_acoustic(0.1, temperature_ratio=10.0, mass_ratio=100.0)
    >>> round(omega.real, 5), round(omega.imag, 5)
    (0.01178, -0.00076)
    """
    k, tau, mu = _floats(k, temperature_ratio, mass_ratio)
    k_start = 0.01
    tau_start = np.maximum(tau, 30.0)

    def parameters(s):
        return k_start * (k / k_start) ** s, tau_start * (tau / tau_start) ** s

    def dielectric(omega, s, derivative):
        kk, tt = parameters(s)
        species = (Maxwellian(), Maxwellian.ions(mass_ratio=mu, temperature_ratio=tt))
        return electrostatic_dielectric(omega, kk, species, derivative=derivative)

    real = np.asarray(ion_acoustic_fluid(k_start, tau_start, mu)).real
    ratio = np.sqrt(np.pi / 8) * (1 / np.sqrt(mu) + tau_start**1.5 * np.exp(-tau_start / 2 - 1.5))
    omega = _track(
        lambda w, s: dielectric(w, s, 0),
        lambda w, s: dielectric(w, s, 1),
        real * (1 - 1j * ratio),
        _path_steps(k / k_start, tau / tau_start),
    )
    return _out(omega)


# ---------------------------------------------------------------------------------------------
# Beams and streams
# ---------------------------------------------------------------------------------------------
def two_stream_cold(k, beam_speed, beam_density=0.5, all_roots=False):
    """Compute the cold symmetric two-stream frequencies in closed form.

    Two cold electron beams of density n_b each at ±v_b over immobile ions:
    1 = n_b/(ω − k v_b)² + n_b/(ω + k v_b)², a quadratic in ω² with the roots
    ω² = k²v_b² + n_b ± √(4 n_b k² v_b² + n_b²). The "slow" root ω² < 0, i.e. the purely growing
    ω = iγ, exists for k² v_b² < 2 n_b; the maximum growth is γ = √n_b/2 at k² v_b² = 3 n_b/4.

    Parameters
    ----------
    k : float or array_like
        The wavenumber in units of 1/λ_De (any length unit L works, with speeds in ω_pe L).
    beam_speed : float or array_like
        v_b, each beam's speed.
    beam_density : float or array_like, optional
        n_b, each beam's density (its ω_pb²). Default: ``0.5``, so the total density is 1.
    all_roots : bool, optional
        Return all four roots instead of the slow one. Default: ``False``.

    Returns
    -------
    complex or numpy.ndarray
        The slow root √(ω₋²) with Im ω ≥ 0: iγ where unstable, the real slow beam mode where
        stable. With ``all_roots``, an extra last axis of length 4:
        (√(ω₊²), −√(ω₊²), √(ω₋²), −√(ω₋²)).

    References
    ----------
    C. K. Birdsall and A. B. Langdon, Plasma Physics via Computer Simulation (IOP, 1991),
    sec. 5.10.

    Examples
    --------
    >>> omega = two_stream_cold(np.sqrt(3 / 8), beam_speed=1.0)
    >>> round(omega.real, 6), round(omega.imag, 6)
    (0.0, 0.353553)
    """
    k = np.asarray(k, dtype=float)
    a2 = (k * np.asarray(beam_speed, dtype=float)) ** 2
    n = np.asarray(beam_density, dtype=float)
    root = np.sqrt(4 * n * a2 + n**2)
    fast = np.sqrt(a2 + n + root + 0j)
    slow = np.sqrt(a2 + n - root + 0j)
    if all_roots:
        return np.stack(np.broadcast_arrays(fast, -fast, slow, -slow), axis=-1)
    return _out(slow)


def beam_plasma_cold(k, beam_speed, beam_density=0.1, plasma_density=1.0, all_roots=False):
    """Compute the cold beam-plasma frequencies: the roots of a quartic.

    A cold electron beam of density n_b at speed v_b through a cold plasma of density n_p
    (immobile ions): 1 = n_p/ω² + n_b/(ω − k v_b)², i.e.
    ω⁴ − 2kv_b ω³ + (k²v_b² − n_p − n_b) ω² + 2kv_b n_p ω − n_p k²v_b² = 0. For n_b ≪ n_p the
    maximum growth is γ ≈ (√3/2)(n_b/(2 n_p))^(1/3) √n_p at k v_b ≈ √n_p.

    Parameters
    ----------
    k : float or array_like
        The wavenumber in units of 1/λ_De (any length unit L works, with speeds in ω_pe L).
    beam_speed : float or array_like
        v_b.
    beam_density : float or array_like, optional
        n_b. Default: ``0.1``.
    plasma_density : float or array_like, optional
        n_p. Default: ``1.0``.
    all_roots : bool, optional
        Return all four roots, sorted by real part. Default: ``False``.

    Returns
    -------
    complex or numpy.ndarray
        The root with the largest imaginary part where it is unstable, else the real root closest
        to the slow beam mode k v_b − √n_b. With ``all_roots``, an extra last axis of length 4.

    References
    ----------
    C. K. Birdsall and A. B. Langdon, Plasma Physics via Computer Simulation (IOP, 1991),
    sec. 5.10.
    R. J. Briggs, Electron-Stream Interaction with Plasmas (MIT Press, 1964).

    Examples
    --------
    >>> omega = beam_plasma_cold(1.0, beam_speed=1.0, beam_density=0.001)
    >>> round(omega.real, 4), round(omega.imag, 4)
    (0.9588, 0.066)
    """
    k, v, nb, npl = _floats(k, beam_speed, beam_density, plasma_density)
    a = k * v
    # companion matrices of the monic quartic ω⁴ + c₃ω³ + c₂ω² + c₁ω + c₀
    c = np.stack([-npl * a**2, 2 * a * npl, a**2 - npl - nb, -2 * a], axis=-1)
    companion = np.zeros(k.shape + (4, 4))
    companion[..., 1:, :-1] = np.eye(3)
    companion[..., :, -1] = -c
    roots = np.linalg.eigvals(companion).astype(complex)
    roots = np.take_along_axis(roots, np.argsort(roots.real, axis=-1), axis=-1)
    if all_roots:
        return roots
    growth = roots.imag.max(axis=-1)
    unstable = growth > 1e-10 * (np.abs(a) + np.sqrt(npl))
    by_growth = np.argmax(roots.imag, axis=-1)
    by_mode = np.argmin(np.abs(roots - (a - np.sqrt(nb))[..., None]), axis=-1)
    chosen = np.take_along_axis(roots, np.where(unstable, by_growth, by_mode)[..., None], axis=-1)[..., 0]
    return _out(np.where(unstable, chosen, chosen.real))


def two_stream(k, beam_speed, thermal_speed, beam_density=0.5):
    """Compute the kinetic purely growing root of two warm counter-streaming electron beams.

    Two Maxwellian electron beams of density n_b and thermal speed v_t at ±v_b over immobile
    ions. By symmetry ε(iγ, k) is real, and the two-stream instability is purely growing,
    ω = iγ: it exists where ε(0, k) < 0, and γ is the largest zero of ε(iγ, k), found by a scan
    and bisection. Where the beams are stable, the least damped root reached by Newton from
    phase speeds between 0 and v_b is returned instead, with Re ω ≥ 0 (−conj ω is a root too):
    past the marginal wavenumber the growing root iγ meets its mirror −iγ near γ = 0 and turns
    into a weakly damped oscillating pair, so γ(k) stays continuous. For v_t → 0 the growth
    approaches :func:`two_stream_cold`.

    Parameters
    ----------
    k : float or array_like
        The wavenumber in units of 1/λ_De (ω(−k) = ω(k)).
    beam_speed : float or array_like
        v_b in units of v_the.
    thermal_speed : float or array_like
        v_t of each beam in units of v_the, positive.
    beam_density : float or array_like, optional
        n_b, each beam's density. Default: ``0.5``, so the total density is 1.

    Returns
    -------
    complex or numpy.ndarray
        ω in units of ω_pe: iγ where unstable; ``nan`` where stable and Newton found no root.

    See Also
    --------
    two_stream_cold : The cold limit.

    References
    ----------
    T. H. Stix, Waves in Plasmas (AIP, 1992), sec. 8.11.
    C. K. Birdsall and A. B. Langdon, Plasma Physics via Computer Simulation (IOP, 1991),
    sec. 5.10.

    Examples
    --------
    >>> omega = two_stream(0.2, beam_speed=3.0, thermal_speed=0.3)
    >>> round(omega.real, 6), round(omega.imag, 4)
    (0.0, 0.3491)
    """
    k, v, vt, n = _floats(k, beam_speed, thermal_speed, beam_density)
    k = np.abs(k)
    species = (
        Maxwellian(density=n, thermal_speed=vt, drift=v),
        Maxwellian(density=n, thermal_speed=vt, drift=-v),
    )

    def dispersion(gamma, axis=False):
        kk = k[..., None] if axis else k
        s = species if not axis else tuple(
            Maxwellian(density=n[..., None], thermal_speed=vt[..., None], drift=d * v[..., None]) for d in (1, -1)
        )
        with np.errstate(all="ignore"):
            value = np.asarray(electrostatic_dielectric(1j * gamma, kk, s)).real
        return np.where(np.isnan(value), np.inf, value)

    unstable = dispersion(np.zeros(k.shape)) < 0
    # unstable: bracket the largest zero of ε(iγ) in (0, top], with ε(i top) > 0 (ε → 1 as γ → ∞)
    top = np.full(k.shape, 0.5 * np.sqrt(2 * n) + 0.1)
    for _ in range(60):
        low = unstable & (dispersion(top) <= 0)
        if not low.any():
            break
        top = np.where(low, 2 * top, top)
    grid = top[..., None] * np.linspace(0.0, 1.0, 257)
    negative = dispersion(grid, axis=True) < 0
    change = negative[..., :-1] & ~negative[..., 1:]
    index = change.shape[-1] - 1 - np.argmax(change[..., ::-1], axis=-1)
    a = np.take_along_axis(grid, index[..., None], axis=-1)[..., 0]
    b = np.take_along_axis(grid, index[..., None] + 1, axis=-1)[..., 0]
    growing = 1j * _bisect(dispersion, np.where(unstable, a, 0.0), np.where(unstable, b, 0.0)) + 0.0
    if unstable.all():
        return _out(growing)
    # stable: the least damped of the roots reached from phase speeds below the beam speed
    kk = k[..., None]
    beams = tuple(
        Maxwellian(density=n[..., None], thermal_speed=vt[..., None], drift=d * v[..., None]) for d in (1, -1)
    )
    slow = np.asarray(two_stream_cold(k, v, n), dtype=complex).real
    speeds = (k * np.abs(v))[..., None] * np.linspace(0.1, 0.9, 5)
    langmuir_like = np.sqrt(2 * n + 3 * (k * vt) ** 2)[..., None]
    guesses = np.concatenate([slow[..., None], speeds, langmuir_like], axis=-1) - 0.01j
    damped = _most_unstable(
        lambda w: electrostatic_dielectric(w, kk, beams),
        lambda w: electrostatic_dielectric(w, kk, beams, derivative=1),
        guesses,
    )
    damped = np.where(damped.real < 0, -np.conj(damped), damped)
    return _out(np.where(unstable, growing, damped))


def bump_on_tail(k, beam_density=0.1, beam_speed=4.5, beam_thermal_speed=0.5, bulk_density=None):
    """Compute the most unstable kinetic root of a bump-on-tail distribution.

    A Maxwellian bulk (density n_0, thermal speed 1, at rest) plus a Maxwellian beam (n_b, v_b,
    v_th,b) over immobile ions. Newton is started from the Bohm–Gross frequency and from phase
    speeds across the beam (k·v for v from v_b − 3 v_th,b to v_b + v_th,b), and the root with the
    largest imaginary part is returned: the growing Langmuir/beam mode where the bump makes the
    distribution unstable (phase speed on its rising flank), the least damped root found where
    not. For k < 0 the roots are those of |k| mirrored, ω → −conj ω.

    Parameters
    ----------
    k : float or array_like
        The wavenumber in units of 1/λ_De, nonzero.
    beam_density : float or array_like, optional
        n_b. Default: ``0.1``.
    beam_speed : float or array_like, optional
        v_b in units of v_the. Default: ``4.5``.
    beam_thermal_speed : float or array_like, optional
        v_th,b in units of v_the, positive. Default: ``0.5``.
    bulk_density : float or array_like, optional
        n_0. Default: ``1 − n_b``, so the total electron density is 1.

    Returns
    -------
    complex or numpy.ndarray
        ω in units of ω_pe; ``nan`` if no guess converged.

    References
    ----------
    T. M. O'Neil and J. H. Malmberg, "Transition of the dispersion roots from beam-type to
    Landau-type solutions", Phys. Fluids 11, 1754 (1968).

    Examples
    --------
    >>> omega = bump_on_tail(0.3)
    >>> round(omega.real, 4), round(omega.imag, 4)
    (1.0012, 0.1981)
    """
    bulk = 1 - np.asarray(beam_density, dtype=float) if bulk_density is None else bulk_density
    k, nb, vb, vtb, n0 = _floats(k, beam_density, beam_speed, beam_thermal_speed, bulk)
    species = (
        Maxwellian(density=n0[..., None]),
        Maxwellian(density=nb[..., None], thermal_speed=vtb[..., None], drift=vb[..., None]),
    )
    fractions = np.linspace(-3.0, 1.0, 9)
    phase_speeds = vb[..., None] + fractions * vtb[..., None]
    guesses = np.concatenate(
        [
            (np.sign(k) * np.sqrt(n0 + nb + 3 * k**2))[..., None] + 0.01j,
            k[..., None] * phase_speeds + 0.01j * np.abs(k[..., None]),
        ],
        axis=-1,
    )
    kk = k[..., None]
    omega = _most_unstable(
        lambda w: electrostatic_dielectric(w, kk, species),
        lambda w: electrostatic_dielectric(w, kk, species, derivative=1),
        guesses,
    )
    return _out(omega)


def maximum_growth(function, k_range, samples=64, tol=1e-8):
    """Find the wavenumber of maximum growth rate of a dispersion relation.

    ``function`` is sampled at ``samples`` points across ``k_range``, and the largest Im ω is
    refined by golden-section search between the neighbours of the best sample.

    Parameters
    ----------
    function : callable
        ``function(k)`` returning the complex ω for an array of wavenumbers, e.g.
        :func:`bump_on_tail` or ``lambda k: two_stream(k, 3.0, 0.3)``.
    k_range : (float, float)
        The interval of wavenumbers searched.
    samples : int, optional
        The number of samples of the initial scan. Default: ``64``.
    tol : float, optional
        The absolute tolerance in k of the refinement. Default: ``1e-8``.

    Returns
    -------
    tuple of (float, complex)
        The wavenumber of maximum growth and ω there.

    Raises
    ------
    ValueError
        If ``function`` returns no finite value on the scan.

    Examples
    --------
    >>> k, omega = maximum_growth(lambda k: two_stream_cold(k, 1.0), (0.01, 1.4))
    >>> round(k, 5), round(omega.imag, 6)   # √(3/8), √0.5/2
    (0.61237, 0.353553)
    """
    k = np.linspace(k_range[0], k_range[1], samples)
    growth = np.asarray(function(k)).imag
    if not np.isfinite(growth).any():
        raise ValueError("the function returned no finite frequency in k_range")
    best = int(np.nanargmax(growth))
    lo, hi = k[max(best - 1, 0)], k[min(best + 1, samples - 1)]

    def rate(x):
        value = np.asarray(function(np.array([x]))).imag[0]
        return value if np.isfinite(value) else -np.inf

    ratio = (np.sqrt(5) - 1) / 2
    x1, x2 = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    f1, f2 = rate(x1), rate(x2)
    while hi - lo > tol:
        if f1 > f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - ratio * (hi - lo)
            f1 = rate(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + ratio * (hi - lo)
            f2 = rate(x2)
    k_best = float((lo + hi) / 2)
    return k_best, complex(np.asarray(function(np.array([k_best])))[0])


# ---------------------------------------------------------------------------------------------
# Weibel
# ---------------------------------------------------------------------------------------------
def weibel(k, anisotropy, parallel_thermal_speed):
    """Compute the purely growing (or damped) root of the electron Weibel instability.

    Transverse electromagnetic waves along k ∥ z in bi-Maxwellian electrons (T⊥ across k, T∥
    along k) with immobile ions:

    ω² − k²c² − ω_pe² + ω_pe² A [1 + ζ Z(ζ)] = 0,  A = T⊥/T∥,  ζ = ω/(√2 k v_∥),

    with v_∥ = √(T∥/m_e). It reduces to light waves ω² = ω_pe² + k²c² for T → 0. The root is
    ω = iγ, with γ > 0 exactly for k²c² < (A − 1) ω_pe² and γ < 0 (a damped, non-oscillating
    mode) beyond; γ is found by bisection of the real function D(iγ), which decreases in γ. Near
    the cutoff γ ≈ √(2/π) k v_∥ (A − 1 − k²c²/ω_pe²)/A.

    **Units:** unlike the rest of this module, k in ω_pe/c, ω in ω_pe and speeds in c.

    Parameters
    ----------
    k : float or array_like
        The wavenumber in units of ω_pe/c (ω(−k) = ω(k), ω(0) = 0).
    anisotropy : float or array_like
        A = T⊥/T∥, positive; unstable for A > 1.
    parallel_thermal_speed : float or array_like
        v_∥ = √(T∥/m_e) in units of c, positive.

    Returns
    -------
    complex or numpy.ndarray
        ω = iγ in units of ω_pe.

    References
    ----------
    E. S. Weibel, "Spontaneously growing transverse waves in a plasma due to an anisotropic
    velocity distribution", Phys. Rev. Lett. 2, 83 (1959).
    R. C. Davidson, D. A. Hammer, I. Haber and C. E. Wagner, "Nonlinear development of
    electromagnetic instabilities in anisotropic plasmas", Phys. Fluids 15, 317 (1972).
    N. A. Krall and A. W. Trivelpiece, Principles of Plasma Physics (McGraw-Hill, 1973), sec. 9.10.

    Examples
    --------
    >>> round(weibel(1.0, anisotropy=4.0, parallel_thermal_speed=0.1).imag, 4)
    0.061
    >>> weibel(np.sqrt(3.0), anisotropy=4.0, parallel_thermal_speed=0.1).imag < 1e-12
    True
    """
    k, a, v = _floats(k, anisotropy, parallel_thermal_speed)
    k = np.abs(k)
    zero = k == 0
    k = np.where(zero, 1.0, k)
    scale = _SQRT2 * k * v

    def dispersion(gamma):
        with np.errstate(all="ignore"):
            w, _ = _response(1j * gamma / scale)
            value = (-(gamma**2) - k**2 - 1 + a * w).real
        return np.where(np.isnan(value), np.inf, value)

    margin = a - 1 - k**2
    hi = np.sqrt(np.maximum(margin, 0.0))
    lo = np.zeros_like(hi)
    # stable: search downwards for D > 0 (D → +∞ as γ → −∞)
    step = np.where(margin < 0, -0.5 * scale, 0.0)
    for _ in range(80):
        need = (margin < 0) & (dispersion(lo) <= 0)
        if not need.any():
            break
        hi = np.where(need, lo, hi)
        lo = np.where(need, lo + step, lo)
        step = 2 * step
    gamma = _bisect(dispersion, lo, hi)
    return _out(np.where(zero, 0.0, 1j * gamma + 0.0))
