"""Arguments that used to be accepted but ignored: data.lineout(x=), data.view(x=, y=, coords=,
plane=), dataset data.scatter(color=), a theory given as points in plot_measured_vs_theory, and
color_by values of the orbit plots other than "classification"."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

import plasma_plots  # noqa: E402, F401
from plasma_plots.plotting import (plot_measured_vs_theory,  # noqa: E402
                                   plot_orbit_grid, plot_orbit_poloidal)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def disk(n1=6, n2=8, n_t=3):
    """A cell-centered annulus: eta2 leaves out the periodic seam, as Struphy's grids do."""
    e1, e2 = np.linspace(0, 1, n1), (np.arange(n2) + 0.5) / n2
    E1, E2 = np.meshgrid(e1, e2, indexing="ij")
    R, TH = 0.2 + 0.8 * E1, 2 * np.pi * E2
    return xr.DataArray(
        np.random.default_rng(0).standard_normal((n_t, n1, n2, 1)),
        dims=("t", "eta1", "eta2", "eta3"),
        coords={
            "t": np.arange(n_t, dtype=float),
            "eta1": e1,
            "eta2": e2,
            "eta3": [0.0],
            "X": (("eta1", "eta2", "eta3"), (R * np.cos(TH))[..., None]),
            "Y": (("eta1", "eta2", "eta3"), (R * np.sin(TH))[..., None]),
            "Z": (("eta1", "eta2", "eta3"), np.zeros((n1, n2, 1))),
        },
        name="n",
    )


def test_data_lineout_checks_x_like_the_plot():
    field = disk()
    profile = field.plasma.data.lineout(x="eta1", t=-1, eta2=0.3, eta3=0)
    assert profile.dims == ("eta1",)
    with pytest.raises(ValueError, match="not the remaining dimension"):
        field.plasma.data.lineout(x="eta2", t=-1, eta2=0.3, eta3=0)
    with pytest.raises(ValueError, match="exactly one remaining dimension"):
        field.plasma.data.lineout(x="eta1", t=-1, eta3=0)


def test_data_view_orders_the_frames_and_closes_the_seam_in_physical_coordinates():
    field = disk()
    frames = field.plasma.data.view(x="eta2", y="eta1", eta3=0)
    assert frames.dims == ("t", "eta2", "eta1")
    physical = field.plasma.data.view(coords="physical", plane="XY", eta3=0)
    assert physical.dims == ("t", "eta1", "eta2")
    assert physical.sizes["eta2"] == field.sizes["eta2"] + 1  # the seam, closed
    np.testing.assert_allclose(physical.isel(eta2=-1), physical.isel(eta2=0))
    # every frame matches the slice the plot draws at that time
    np.testing.assert_allclose(
        physical.isel(t=1),
        field.plasma.data.slice(coords="physical", plane="XY", t=1, eta3=0),
    )
    with pytest.raises(ValueError, match="expected 't'"):
        field.plasma.data.view(x="eta1", y="eta2")  # eta3 left over
    with pytest.raises(ValueError, match="physical coordinates"):
        field.drop_vars(["X", "Y", "Z"]).plasma.data.view(coords="physical", eta3=0)


def test_dataset_data_scatter_returns_what_the_plot_draws():
    t = np.arange(3.0)
    markers = xr.Dataset(
        {
            "x": (("t", "marker"), np.arange(12.0).reshape(3, 4)),
            "y": (("t", "marker"), np.ones((3, 4))),
            "density": (("t", "marker"), 10 + np.arange(12.0).reshape(3, 4)),
            "weight": (("t", "marker"), np.ones((3, 4))),
        },
        coords={"t": t, "marker": np.arange(4)},
    )
    selected = markers.plasma.data.scatter(x="x", y="y", color="density", t=-1)
    assert set(selected.data_vars) == {"x", "y", "density"}
    np.testing.assert_allclose(selected.density, markers.density.isel(t=-1))
    initial = markers.plasma.data.scatter(x="x", y="y", color="x", color_at=0, t=-1)
    assert set(initial.data_vars) == {"x", "y", "color"}
    np.testing.assert_allclose(
        initial.color, markers.x.isel(t=0)
    )  # colored by the start
    np.testing.assert_allclose(initial.x, markers.x.isel(t=-1))
    assert set(markers.plasma.data.scatter(x="x", y="y", t=0).data_vars) == {"x", "y"}
    with pytest.raises(ValueError, match="not data variables"):
        markers.plasma.data.scatter(x="x", y="y", color="speed", t=0)


def test_measured_against_a_theory_given_as_points():
    k = np.array([1.0, 2.0, 3.0, 5.0])
    measured = xr.DataArray(1.1 * k, dims="k", coords={"k": k}, name="omega")
    theory_points = (
        np.array([0.0, 4.0]),
        np.array([0.0, 4.0]),
    )  # omega = k, only up to k = 4
    result = plot_measured_vs_theory(measured, {"linear": theory_points})
    assert len(result.ax) == 2
    errors = result.ax[1].lines[0].get_ydata()
    np.testing.assert_allclose(errors[:3], 0.1)
    assert np.isnan(errors[3])  # outside the theory's points


def orbits(n_t=60, n=6):
    t = np.linspace(0, 30, n_t)[:, None]
    theta = np.linspace(0, 2 * np.pi, n)[None] + 0.3 * t
    v_par = np.where(np.arange(n)[None] % 2 == 0, np.cos(0.4 * t), 1.0 + 0 * t)
    return xr.Dataset(
        {
            "x": (("t", "marker"), (3.0 + 0.5 * np.cos(theta)) * np.ones_like(t)),
            "y": (("t", "marker"), np.zeros((n_t, n))),
            "z": (("t", "marker"), 0.5 * np.sin(theta)),
            "v_par": (("t", "marker"), v_par),
        },
        coords={"t": t[:, 0], "marker": np.arange(n)},
    )


def test_orbit_plots_color_by_a_quantity_and_reject_unknown_values():
    paths = orbits()
    by_v_par = plot_orbit_poloidal(paths, color_by="v_par")
    lines = [a for a in by_v_par.artists if isinstance(a, LineCollection)]
    assert len(lines) == paths.sizes["marker"]
    assert len(by_v_par.fig.axes) == 2  # with a color bar
    lo, hi = lines[0].norm.vmin, lines[0].norm.vmax
    assert lo == pytest.approx(float(paths.v_par.min())) and hi == pytest.approx(
        float(paths.v_par.max())
    )
    grid = plot_orbit_grid(paths, markers=4, ncols=2, color_by="t")
    assert sum(isinstance(a, LineCollection) for a in grid.artists) == 4
    assert any(
        "trapped" in ax.get_title() for ax in grid.ax.ravel()
    )  # titles still name the class
    assert len(paths.plasma.plot.poloidal(color_by=None).fig.axes) == 1
    with pytest.raises(ValueError, match="color_by must be"):
        plot_orbit_poloidal(paths, color_by="speed")
    with pytest.raises(ValueError, match="color_by must be"):
        plot_orbit_grid(paths, color_by="classificaton")


def test_dispersion_takes_theory_dicts_complex_branches_and_struphy_style_objects():
    from plasma_plots.plotting import plot_dispersion

    t = np.linspace(0, 40, 200)
    x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    field = xr.DataArray(
        np.cos(2 * x[None] - 2 * t[:, None]),
        dims=("t", "eta1"),
        coords={"t": t, "eta1": x},
    )

    def waves(k):  # like the theory functions: a dict of complex branches
        return {"fast": 1.0 * k + 0.01j, "slow": 0.5 * k}

    class StruphyLike:  # like struphy.dispersion_relations objects: callable, returns a dict
        def __call__(self, k):
            return {"light wave": k}

    for branches, labels in (
        (waves, {"fast", "slow"}),
        ({"MHD": waves}, {"MHD: fast", "MHD: slow"}),
        (StruphyLike(), {"light wave"}),
        ({"c": lambda k: k + 0j}, {"c"}),
    ):
        result = plot_dispersion(field, branches=branches)
        shown = {text.get_text() for text in result.ax.get_legend().get_texts()}
        assert labels <= shown
        for line in result.ax.lines:
            assert np.isrealobj(line.get_ydata())
    from plasma_plots.analysis import power_spectrum
    from plasma_plots.spectral import trace_branch

    traced = trace_branch(
        power_spectrum(field), lambda k: 1.0 * k + 0.05j, k_range=(1.5, 2.5)
    ).dropna("k")
    assert traced.k.values.tolist() == [2.0]
    result = plot_measured_vs_theory(traced.omega, lambda k: k + 0.2j)
    np.testing.assert_allclose(
        result.ax[1].lines[0].get_ydata(), traced.relative_error, atol=1e-12
    )
