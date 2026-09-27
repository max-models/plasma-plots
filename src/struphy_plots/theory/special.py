"""Special functions of plasma theory: the Faddeeva function, the plasma dispersion function and
complete elliptic integrals, in plain numpy (no scipy needed).
"""

from __future__ import annotations

import numpy as np

_SQRT_PI = np.sqrt(np.pi)


def _out(values):
    """A Python scalar for scalar input, else the array (the convention of the theory modules)."""
    values = np.asarray(values)
    return values.item() if values.ndim == 0 else values


def _weideman_coefficients(n: int = 64):
    """Coefficients of Weideman's rational approximation of the Faddeeva function."""
    m = 2 * n
    k = np.arange(-m + 1, m)
    length = np.sqrt(n / np.sqrt(2))
    t = length * np.tan(k * np.pi / (2 * m))
    f = np.concatenate([[0.0], np.exp(-(t**2)) * (length**2 + t**2)])
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / (2 * m)
    return length, np.flipud(a[1 : n + 1])


_LENGTH, _COEFFICIENTS = _weideman_coefficients()


def faddeeva(z):
    """Compute the Faddeeva function w(z) = exp(−z²) erfc(−iz).

    Weideman's rational approximation with 64 terms in the upper half plane (relative accuracy
    about 1e-13), continued to the lower half plane with w(z) = 2 exp(−z²) − w(−z).

    Parameters
    ----------
    z : complex or array_like
        The argument.

    Returns
    -------
    complex or numpy.ndarray
        w(z), with the shape of ``z``.

    References
    ----------
    J. A. C. Weideman, "Computation of the complex error function", SIAM J. Numer. Anal. 31,
    1497 (1994).

    Examples
    --------
    >>> round(faddeeva(0.0).real, 12), round(faddeeva(1j).real, 6)   # erfc(1) e¹ at z = i
    (1.0, 0.427584)
    """
    z = np.asarray(z, dtype=complex)
    lower = z.imag < 0
    upper = np.where(lower, -z, z)
    denominator = _LENGTH - 1j * upper
    ratio = (_LENGTH + 1j * upper) / denominator
    w = 2 * np.polyval(_COEFFICIENTS, ratio) / denominator**2 + 1 / (_SQRT_PI * denominator)
    with np.errstate(over="ignore", invalid="ignore"):
        w = np.where(lower, 2 * np.exp(-(z**2)) - w, w)
    return _out(w)


def plasma_dispersion(zeta, derivative: int = 0):
    """Compute the plasma dispersion function Z(ζ) of Fried and Conte, or its first derivative.

    Z(ζ) = i √π w(ζ) is the analytic continuation of (1/√π) ∫ exp(−x²)/(x − ζ) dx from the upper
    half plane (Landau's prescription), and Z'(ζ) = −2 [1 + ζ Z(ζ)]. For a Maxwellian
    f ∝ exp(−v²/(2 v_th²)) the argument is ζ = ω/(√2 k v_th).

    Parameters
    ----------
    zeta : complex or array_like
        The argument ζ.
    derivative : {0, 1}, optional
        0 for Z(ζ), 1 for Z'(ζ). Default: ``0``.

    Returns
    -------
    complex or numpy.ndarray
        Z(ζ) or Z'(ζ), with the shape of ``zeta``.

    Raises
    ------
    ValueError
        If ``derivative`` is not 0 or 1.

    References
    ----------
    B. D. Fried and S. D. Conte, The Plasma Dispersion Function (Academic Press, 1961).

    Examples
    --------
    >>> round(plasma_dispersion(0.0).imag, 6)   # i √π at the origin
    1.772454
    >>> plasma_dispersion(0.0, derivative=1)
    (-2+0j)
    """
    if derivative not in (0, 1):
        raise ValueError(f"derivative must be 0 or 1; got {derivative!r}")
    zeta = np.asarray(zeta, dtype=complex)
    z = 1j * _SQRT_PI * faddeeva(zeta)
    return _out(z if derivative == 0 else -2 * (1 + zeta * z))


def elliptic_k(m):
    """Compute the complete elliptic integral of the first kind K(m), with parameter m = k².

    K(m) = ∫₀^{π/2} dθ / √(1 − m sin²θ), by the arithmetic-geometric mean.

    Parameters
    ----------
    m : float or array_like
        The parameter, m < 1 (K diverges logarithmically as m → 1).

    Returns
    -------
    float or numpy.ndarray
        K(m); ``inf`` at m = 1, ``nan`` for m > 1.

    Examples
    --------
    >>> round(float(elliptic_k(0.0)), 12)   # π/2
    1.570796326795
    """
    m = np.asarray(m, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        a, b = np.ones_like(m), np.sqrt(1 - m)
        for _ in range(40):
            a, b = (a + b) / 2, np.sqrt(a * b)
        return _out(np.where(m == 1, np.inf, np.pi / (2 * a)))


def elliptic_e(m):
    """Compute the complete elliptic integral of the second kind E(m), with parameter m = k².

    E(m) = ∫₀^{π/2} √(1 − m sin²θ) dθ, by the arithmetic-geometric mean.

    Parameters
    ----------
    m : float or array_like
        The parameter, m ≤ 1.

    Returns
    -------
    float or numpy.ndarray
        E(m); ``nan`` for m > 1.

    Examples
    --------
    >>> round(float(elliptic_e(1.0)), 12)
    1.0
    """
    m = np.asarray(m, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        a, b = np.ones_like(m), np.sqrt(1 - m)
        c2 = m.copy()
        total = c2 / 2
        power = 0.5
        for _ in range(40):
            c = (a - b) / 2
            a, b = (a + b) / 2, np.sqrt(a * b)
            power *= 2
            total = total + power * c**2  # E = K (1 − Σ 2ⁿ⁻¹ cₙ²), c₀² = m
        e = elliptic_k(m) * (1 - total)
        return _out(np.where(m == 1, 1.0, e))
