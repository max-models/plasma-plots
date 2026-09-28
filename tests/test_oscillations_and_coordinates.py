"""oscillation_frequency, map_coordinate and plot.convergence."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import plasma_plots  # noqa: E402, F401
from plasma_plots.analysis import oscillation_frequency  # noqa: E402
from plasma_plots.arrays import axis_label, map_coordinate  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def damped(omega=1.3, gamma=0.05, n=400):
    t = np.linspace(0, 40, n)
    return xr.DataArray(np.exp(-gamma * t) * np.cos(omega * t + 0.4), dims="t", coords={"t": t}, name="probe")


@pytest.mark.parametrize("method", ["zero_crossings", "peaks"])
def test_the_frequency_of_a_damped_oscillation_is_found_to_a_fraction_of_a_sample(method):
    fit = damped().plasma.analysis.oscillation_frequency(method=method)
    assert fit.omega == pytest.approx(1.3, rel=1e-3) and fit.period == pytest.approx(2 * np.pi / 1.3, rel=1e-3)
    assert fit.method == method and len(fit.times) >= 8


def test_a_window_and_the_mean_are_respected():
    signal = damped() + 5.0  # crossings of the mean, not of zero
    fit = oscillation_frequency(signal, window=(10.0, 30.0))
    assert fit.omega == pytest.approx(1.3, rel=2e-3) and fit.times.min() >= 10 and fit.times.max() <= 30
    assert oscillation_frequency(signal, detrend=False) is None  # never crosses zero
    energy = damped() ** 2
    assert oscillation_frequency(energy, method="peaks", detrend=False).omega == pytest.approx(2.6, rel=2e-3)


def test_bad_input_is_refused():
    with pytest.raises(ValueError, match="method"):
        oscillation_frequency(damped(), method="fourier")
    with pytest.raises(ValueError, match="dims"):
        oscillation_frequency(damped().expand_dims(x=[0.0]))
    assert oscillation_frequency(damped().isel(t=slice(0, 3))) is None


def profile():
    eta1, t = np.linspace(0, 1, 11), np.array([0.0, 1.0])
    return xr.DataArray(np.outer(t + 1, eta1), dims=("t", "eta1"), coords={"t": t, "eta1": eta1}, name="T")


def test_a_mapped_coordinate_renames_the_dimension_and_labels_the_plots():
    mapped = profile().plasma.analysis.map_coordinate("eta1", lambda e: 0.1 + 0.9 * e, name="r", units="m",
                                                      label="$r$")
    assert mapped.dims == ("t", "r") and "eta1" not in mapped.coords
    np.testing.assert_allclose(mapped.r, 0.1 + 0.9 * profile().eta1.values)
    assert axis_label(mapped, "r") == "$r$ [m]"
    result = mapped.plasma.plot.lineout(x="r", t=-1)
    assert result.ax.get_xlabel() == "$r$ [m]"
    assert float(mapped.sel(r=0.55, t=1.0)) == pytest.approx(2 * 0.5)


def test_a_factor_scales_a_coordinate_in_place():
    scaled = map_coordinate(profile(), "eta1", 2 * np.pi, units="m")
    assert scaled.dims == ("t", "eta1") and scaled.eta1.values[-1] == pytest.approx(2 * np.pi)
    assert axis_label(scaled, "eta1") == "eta1 [m]"
    with pytest.raises(KeyError):
        map_coordinate(profile(), "eta2", 2.0)


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_convergence_plots_several_series_with_their_orders(backend):
    if backend == "plotly":
        pytest.importorskip("plotly")
    n = np.array([16, 32, 64, 128])
    l2 = xr.DataArray(3.0 * n**-2.0, dims="n", coords={"n": n}, name="L2 error")
    linf = xr.DataArray(5.0 * n**-1.0, dims="n", coords={"n": n}, name="max error")
    result = l2.plasma.plot.convergence(linf, backend=backend)
    if backend == "matplotlib":
        labels = [line.get_label() for line in result.ax.lines]
        assert labels[0] == "L2 error" and labels[1].startswith("fit: order -2.00")
        assert labels[3].startswith("fit: order -1.00") and result.ax.get_xscale() == "log"
        return
    names = [trace.name for trace in result.fig.data]
    assert names[1].startswith("fit: order -2.00") and result.fig.layout.xaxis.type == "log"
    with pytest.raises(ValueError, match="one-dimensional"):
        l2.expand_dims(m=[1, 2]).plasma.plot.convergence()
