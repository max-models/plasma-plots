"""Interactive Plotly figures: array.struphy.plotly and struphy_plots.plotly_plots."""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("plotly")

import struphy_plots  # noqa: E402, F401
from struphy_plots.analysis import fit_dispersion_branches, power_spectrum  # noqa: E402
from struphy_plots.plotly_plots import dispersion, space_time  # noqa: E402


@pytest.fixture
def waves():
    """Broadband light waves, omega = |k|, in both directions along z."""
    rng = np.random.default_rng(0)
    t = np.arange(400) * 0.05
    z = np.linspace(0.0, 20.0, 128, endpoint=False)
    T, Z = np.meshgrid(t, z, indexing="ij")
    values = np.zeros_like(T)
    for n in range(1, 40):
        k = 2 * np.pi * n / 20
        for sign in (1, -1):
            values += rng.normal() * np.cos(k * Z + sign * k * T + rng.uniform(0, 2 * np.pi))
    return xr.DataArray(values, dims=("t", "z"), coords={"t": t, "z": z}, name="E_x")


def test_space_time_is_symmetric_about_zero_and_labels_its_axes(waves):
    figure = space_time(waves, title="E_x(z, t)")
    heatmap = figure.data[0]
    assert heatmap.zmin == -heatmap.zmax == -float(np.abs(waves).max())
    assert np.asarray(heatmap.z).shape == (waves.sizes["t"], waves.sizes["z"])
    assert figure.layout.xaxis.title.text == "z" and figure.layout.yaxis.title.text == "t"
    assert figure.layout.title.text == "E_x(z, t)"


def test_space_time_needs_exactly_t_and_one_space_dimension(waves):
    with pytest.raises(ValueError, match="space is required"):
        space_time(waves.expand_dims(y=[0.0, 1.0]))
    with pytest.raises(ValueError, match="select every dimension"):
        space_time(waves.expand_dims(y=[0.0, 1.0]), space="z")


def test_dispersion_draws_the_positive_quadrant_with_branches_and_fits(waves):
    spectrum = power_spectrum(waves, dim="z")
    fits = fit_dispersion_branches(spectrum, n_branches=1)
    figure = dispersion(spectrum, branches={"light": lambda k: k}, fits=fits, dynamic_range=12)
    heatmap, light, fit = figure.data
    assert np.min(heatmap.x) >= 0 and np.min(heatmap.y) >= 0
    assert heatmap.zmax == 0 and heatmap.zmin == -12 and np.nanmax(heatmap.z) == 0
    np.testing.assert_allclose(light.y, light.x)
    assert fit.name.startswith("fit, v = ") and abs(fits[0].velocity - 1) < 0.02
    assert list(figure.layout.xaxis.range) == [0.0, float(spectrum.k.max())]


def test_the_accessor_draws_the_same_figures(waves):
    assert waves.struphy.plotly.space_time().to_dict() == space_time(waves).to_dict()
    spectrum = power_spectrum(waves, dim="z")
    assert spectrum.struphy.plotly.dispersion(omega_max=5).to_dict() == dispersion(spectrum, omega_max=5).to_dict()


def test_dispersion_needs_omega_and_k(waves):
    with pytest.raises(ValueError, match="omega"):
        dispersion(waves)
