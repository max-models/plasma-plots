"""Tests for fit_dispersion_branches: recovering measured wave speeds from a power spectrum."""

import numpy as np
import pytest
import xarray as xr

import plasma_plots  # noqa: F401  (registers .plasma on DataArray/Dataset)
from plasma_plots.analysis import BranchFit, fit_dispersion_branches, power_spectrum


def multi_branch_field(
    velocities,
    *,
    n_modes=32,
    nt=512,
    nx=128,
    tmax=40.0,
    length=2 * np.pi,
    seed=0,
):
    """A broadband (t, eta3) field exciting a straight branch omega = +-v*k for each of
    ``velocities``, at every one of the first ``n_modes`` spatial Fourier modes, with an
    independent random phase per mode and direction -- similar in spirit to noise-driven,
    bidirectionally propagating waves in a real simulation, but built directly in Fourier
    space so each branch's velocity is known exactly.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, tmax, nt)
    x = np.linspace(0.0, length, nx, endpoint=False)
    k_modes = 2 * np.pi * np.fft.rfftfreq(nx, d=length / nx)[1 : n_modes + 1]
    X, T = np.meshgrid(x, t)
    K = k_modes[:, None, None]
    values = np.zeros_like(X)
    for v in velocities:
        for sign in (+1.0, -1.0):
            phase = rng.uniform(0.0, 2 * np.pi, size=(len(k_modes), 1, 1))
            values += np.cos(K * X[None] + sign * v * K * T[None] + phase).sum(axis=0)
    return xr.DataArray(
        values,
        dims=("t", "eta3"),
        coords={"t": t, "eta3": x},
        name="field",
        attrs={"label": "field"},
    )


def test_fit_dispersion_branches_recovers_a_single_branch_velocity():
    spectrum = power_spectrum(multi_branch_field([1.0]))
    (branch,) = fit_dispersion_branches(
        spectrum, n_branches=1, noise_level=0.3, order=3
    )
    assert isinstance(branch, BranchFit)
    assert branch.velocity == pytest.approx(1.0, rel=0.01)
    assert branch.k.shape == branch.omega.shape
    assert branch.k.size > 5


def test_fit_dispersion_branches_separates_two_branches_by_increasing_omega():
    spectrum = power_spectrum(multi_branch_field([0.9, 2.2]))
    slow, fast = fit_dispersion_branches(
        spectrum, n_branches=2, k_range=(5.0, 15.0), noise_level=0.3, order=3
    )
    assert slow.velocity == pytest.approx(0.9, rel=0.02)
    assert fast.velocity == pytest.approx(2.2, rel=0.02)


def test_fit_dispersion_branches_requires_omega_and_k_dims():
    not_a_spectrum = xr.DataArray(np.ones((3, 4)), dims=("t", "eta3"))
    with pytest.raises(ValueError, match="omega.*k"):
        fit_dispersion_branches(not_a_spectrum, n_branches=1)


def test_fit_dispersion_branches_requires_a_positive_branch_count():
    spectrum = power_spectrum(multi_branch_field([1.0]))
    with pytest.raises(ValueError, match="n_branches must be positive"):
        fit_dispersion_branches(spectrum, n_branches=0)


def test_fit_dispersion_branches_raises_when_nothing_matches():
    spectrum = power_spectrum(multi_branch_field([1.0]))
    with pytest.raises(ValueError, match="no k in"):
        fit_dispersion_branches(spectrum, n_branches=3, noise_level=0.3, order=3)


def test_accessor_fit_branches_matches_the_function():
    field = multi_branch_field([1.0])
    spectrum = field.plasma.analysis.dispersion()
    via_accessor = spectrum.plasma.analysis.fit_branches(
        n_branches=1, noise_level=0.3, order=3
    )
    via_function = fit_dispersion_branches(
        spectrum, n_branches=1, noise_level=0.3, order=3
    )
    assert via_accessor[0].velocity == pytest.approx(via_function[0].velocity)
