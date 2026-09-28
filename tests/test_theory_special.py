"""The special functions of plasma_plots.theory, in plain numpy: exact values, identities, and
agreement with scipy where it is installed."""

import numpy as np
import pytest

from plasma_plots.theory.special import elliptic_e, elliptic_k, faddeeva, plasma_dispersion


def test_exact_values_and_identities():
    assert faddeeva(0.0) == pytest.approx(1.0)
    assert plasma_dispersion(0.0) == pytest.approx(1j * np.sqrt(np.pi))
    assert plasma_dispersion(0.0, derivative=1) == pytest.approx(-2.0)
    z = np.array([0.3 + 0.2j, -2.0 + 0.5j, 1.5 - 0.7j, 8.0 + 0.01j])
    # Z'(ζ) = −2(1 + ζZ(ζ)), and numerically as a derivative
    h = 1e-6
    numeric = (plasma_dispersion(z + h) - plasma_dispersion(z - h)) / (2 * h)
    np.testing.assert_allclose(plasma_dispersion(z, derivative=1), numeric, rtol=1e-7)
    # the asymptotic expansion on the real axis, where the Landau term exp(−ζ²) is negligible
    x = np.array([30.0, 50.0])
    np.testing.assert_allclose(plasma_dispersion(x).real, -1 / x - 1 / (2 * x**3) - 3 / (4 * x**5) - 15 / (8 * x**7), rtol=1e-10)
    assert elliptic_k(0.0) == pytest.approx(np.pi / 2)
    assert elliptic_e(0.0) == pytest.approx(np.pi / 2)
    assert elliptic_e(1.0) == pytest.approx(1.0)
    assert np.isinf(elliptic_k(1.0))
    assert faddeeva(np.array([[0.0, 1j]])).shape == (1, 2)
    with pytest.raises(ValueError, match="derivative"):
        plasma_dispersion(0.0, derivative=2)


def test_agrees_with_scipy():
    special = pytest.importorskip("scipy.special")
    x = np.linspace(-30, 30, 241)
    y = np.concatenate([-np.logspace(-3, np.log10(5), 30)[::-1], [0.0], np.logspace(-3, 1.5, 40)])
    z = x[:, None] + 1j * y[None]
    reference = special.wofz(z)
    finite = np.abs(reference) < 1e150
    np.testing.assert_allclose(faddeeva(z)[finite], reference[finite], rtol=5e-13)
    m = np.linspace(0.0, 0.999999, 500)
    np.testing.assert_allclose(elliptic_k(m), special.ellipk(m), rtol=1e-13)
    np.testing.assert_allclose(elliptic_e(m), special.ellipe(m), rtol=1e-13)
