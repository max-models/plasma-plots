"""Volume integrals and field energies, energy budgets, profiles, orbit plots, color limits and
VTK export (the tools taken over from struphy's TAE_example_Shrut branch)."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

import plasma_plots  # noqa: E402, F401
from plasma_plots import spectral as sp  # noqa: E402
from plasma_plots.analysis import (field_energy,  # noqa: E402
                                   quadrature_weights, volume_integral)
from plasma_plots.plotting import (View, _slice_data,  # noqa: E402
                                   color_limits, plot_energy_budget,
                                   plot_orbit_poloidal, plot_orbit_quantities,
                                   plot_profiles)

R0 = 3.0


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def hollow_torus(n1=24, n2=48, n3=48, cell_centred=False):
    """r = 0.2 + 0.8 eta1, theta = 2 pi eta2, phi = 2 pi eta3, with its analytic Jacobian."""
    if cell_centred:
        eta1, eta2, eta3 = ((np.arange(n) + 0.5) / n for n in (n1, n2, n3))
    else:
        eta1, eta2, eta3 = (
            np.linspace(0, 1, n1),
            np.linspace(0, 1, n2),
            np.linspace(0, 1, n3),
        )
    E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    r, theta, phi = 0.2 + 0.8 * E1, 2 * np.pi * E2, 2 * np.pi * E3
    R = R0 + r * np.cos(theta)
    X, Y, Z = R * np.cos(phi), R * np.sin(phi), r * np.sin(theta)
    jac = np.array(
        [
            [
                0.8 * np.cos(theta) * np.cos(phi),
                -2 * np.pi * r * np.sin(theta) * np.cos(phi),
                -2 * np.pi * R * np.sin(phi),
            ],
            [
                0.8 * np.cos(theta) * np.sin(phi),
                -2 * np.pi * r * np.sin(theta) * np.sin(phi),
                2 * np.pi * R * np.cos(phi),
            ],
            [0.8 * np.sin(theta), 2 * np.pi * r * np.cos(theta), 0 * E1],
        ]
    )
    coords = {
        "eta1": eta1,
        "eta2": eta2,
        "eta3": eta3,
        **{n: (("eta1", "eta2", "eta3"), c) for n, c in zip("XYZ", (X, Y, Z))},
    }
    return coords, jac, (X, Y, Z)


VOLUME = 2 * np.pi * R0 * np.pi * (1.0**2 - 0.2**2)


def scalar_field(coords, values, name="f"):
    return xr.DataArray(
        values,
        dims=("eta1", "eta2", "eta3"),
        coords=coords,
        name=name,
        attrs={"label": name},
    )


def vector_field(coords, values, name="u"):
    return xr.DataArray(
        values,
        dims=("component", "eta1", "eta2", "eta3"),
        coords={"component": [0, 1, 2], **coords},
        name=name,
    )


def test_quadrature_weights():
    assert quadrature_weights([0.5]).tolist() == [1.0]
    centres = (np.arange(8) + 0.5) / 8
    np.testing.assert_allclose(quadrature_weights(centres), 1 / 8)
    np.testing.assert_allclose(
        quadrature_weights(np.linspace(0, 1, 5)), [0.125, 0.25, 0.25, 0.25, 0.125]
    )


@pytest.mark.parametrize("cell_centred", [False, True])
def test_volume_integral_of_a_hollow_torus(cell_centred):
    coords, jac, _ = hollow_torus(cell_centred=cell_centred)
    ones = scalar_field(coords, np.ones((24, 48, 48)))
    assert float(volume_integral(ones)) == pytest.approx(VOLUME, rel=2e-3)
    # a density (3-form) is integrated without the volume element
    sqrt_g = np.abs(np.linalg.det(np.moveaxis(jac, (0, 1), (-2, -1))))
    assert float(volume_integral(ones.copy(data=sqrt_g), form=3)) == pytest.approx(
        VOLUME, rel=2e-3
    )


def test_field_energy_is_the_same_in_every_representation():
    coords, jac, (X, Y, Z) = hollow_torus()
    cartesian = np.stack([-Y, X, 0 * X]) / np.hypot(X, Y)  # a unit toroidal field
    inverse = np.linalg.inv(np.moveaxis(jac, (0, 1), (-2, -1)))
    contravariant = np.einsum("...ia,a...->i...", inverse, cartesian)
    sqrt_g = np.abs(np.linalg.det(np.moveaxis(jac, (0, 1), (-2, -1))))
    two_form = sqrt_g * contravariant
    one_form = np.einsum("ai...,a...->i...", jac, cartesian)
    expected = 0.5 * VOLUME
    for values, form in (
        (cartesian, None),
        (contravariant, "v"),
        (two_form, 2),
        (one_form, 1),
    ):
        energy = float(field_energy(vector_field(coords, values), form=form))
        assert energy == pytest.approx(expected, rel=5e-3), form
    weighted = field_energy(
        vector_field(coords, cartesian),
        weight=np.full((24, 48, 48), 2.0),
        normalization=0.5,
    )
    assert float(weighted) == pytest.approx(expected, rel=5e-3)
    with pytest.raises(ValueError, match="vector components"):
        field_energy(scalar_field(coords, X), form=2)
    with pytest.raises(ValueError, match="form"):
        field_energy(scalar_field(coords, X), form=5)


def test_field_energy_keeps_time_and_takes_explicit_quadrature():
    coords, _, _ = hollow_torus(n1=6, n2=12, n3=12)
    values = np.stack([np.ones((6, 12, 12)) * a for a in (1.0, 2.0)])
    field = xr.DataArray(
        values, dims=("t", "eta1", "eta2", "eta3"), coords={"t": [0.0, 1.0], **coords}
    )
    energy = field_energy(field)
    assert energy.dims == ("t",)
    assert float(energy[1] / energy[0]) == pytest.approx(4.0)
    unit = {
        d: np.full(field.sizes[d], 1.0 / field.sizes[d])
        for d in ("eta1", "eta2", "eta3")
    }
    assert field_energy(field, quadrature=unit).dims == ("t",)
    with pytest.raises(ValueError, match="quadrature weights"):
        field_energy(field, quadrature={"eta1": [1.0]})


def test_color_limits_symmetric_and_robust():
    values = np.concatenate([np.linspace(-1, 3, 1000), [100.0]])
    assert color_limits(values) == (-1.0, 100.0)
    assert color_limits(values, symmetric=True) == (-100.0, 100.0)
    lo, hi = color_limits(values, robust=True)
    assert hi < 4 and lo > -1.1
    lo, hi = color_limits(values, symmetric=True, robust=True)
    assert lo == -hi and hi < 4


def test_physical_slices_close_the_periodic_seam_of_cell_centred_grids():
    coords, _, (X, _, _) = hollow_torus(n1=6, n2=16, n3=4, cell_centred=True)
    field = scalar_field(coords, X)
    selected, (xg, yg, _, _) = _slice_data(
        field.isel(eta3=0), View(x="eta1", y="eta2", coordinates="physical", plane="RZ")
    )
    assert selected.sizes["eta2"] == 17
    np.testing.assert_allclose(xg[:, -1], xg[:, 0])  # the seam is closed
    logical, _ = _slice_data(field.isel(eta3=0), View(x="eta1", y="eta2"))
    assert logical.sizes["eta2"] == 16  # logical plots are unchanged
    result = field.plasma.plot.slice(
        x="eta1", y="eta2", coords="physical", plane="RZ", eta3=0, symmetric=True
    )
    lo, hi = result.artists[0].get_clim()
    assert lo == -hi


def test_energy_budget_with_groups():
    t = np.linspace(0, 10, 101)
    transfer = 0.1 * (1 - np.exp(-t))
    scalars = xr.Dataset(
        {
            "en_U": ("t", 0.5 + transfer / 2),
            "en_B": ("t", 0.5 + transfer / 2),
            "en_fv": ("t", 1.0 - transfer),
            "en_B_eq": ("t", 100 + 0 * t),
            "en_tot": ("t", 2.0 + 1e-6 * t),
        },
        coords={"t": t},
    )
    result = plot_energy_budget(
        scalars, groups={"wave": ["en_U", "en_B"], "ions": ["en_fv"]}
    )
    assert len(result.ax) == 3
    labels = [line.get_label() for line in result.ax[0].lines]
    assert labels == [
        "en_U",
        "en_B",
        "en_fv",
        "en_tot",
    ]  # the _eq scalar is left out by default
    wave, ions, minus_ions = result.ax[2].lines[:3]
    np.testing.assert_allclose(wave.get_ydata(), minus_ions.get_ydata())
    assert len(plot_energy_budget(scalars, total=None).ax) == 1


def test_profiles_at_several_times_against_a_mapped_radius():
    t = np.linspace(0, 1, 11)
    eta1 = np.linspace(0, 1, 20)
    field = xr.DataArray(
        np.outer(1 + t, np.sin(np.pi * eta1)),
        dims=("t", "eta1"),
        coords={"t": t, "eta1": eta1},
        name="u",
    )
    result = plot_profiles(field, x="eta1", x_of=lambda e: 0.1 + 0.9 * e)
    assert len(result.artists) == 4
    np.testing.assert_allclose(result.artists[0].get_xdata()[[0, -1]], [0.1, 1.0])
    cube = field.expand_dims(eta2=[0.0, 0.5]).transpose("t", "eta1", "eta2")
    assert len(cube.plasma.plot.profiles(x="eta1", at=[0, 0.5], eta2=0.5).artists) == 2
    mixed = cube.plasma.plot.profiles(
        x="eta1", at=[0, 0.5, -1], eta2=0.5
    )  # positions and a value
    assert mixed.artists[-1].get_label().endswith(f"{float(cube.t[-1]):.3g}")


def orbits_dataset():
    t = np.linspace(0, 40, 200)[:, None]
    rng = np.random.default_rng(0)
    n = 12
    r0, th0 = rng.uniform(0.2, 0.8, n), rng.uniform(0, 2 * np.pi, n)
    trapped = np.arange(n) % 3 == 0
    v_par = np.where(trapped, np.cos(0.3 * t + th0), 1.0 + 0 * t)
    theta = th0 + np.where(trapped, 0.8 * np.sin(0.3 * t), 0.3 * t)
    x, y, z = (
        (R0 + r0 * np.cos(theta)) * np.cos(0.2 * t),
        (R0 + r0 * np.cos(theta)) * np.sin(0.2 * t),
        r0 * np.sin(theta),
    )
    mu = 0.5 + 1e-6 * t + 0 * r0
    x[150:, 1] = y[150:, 1] = z[150:, 1] = v_par[150:, 1] = mu[150:, 1] = 0.0  # lost
    names = ("x", "y", "z", "v_par", "mu")
    return xr.Dataset(
        {
            name: (("t", "marker"), value)
            for name, value in zip(names, (x, y, z, v_par, mu))
        },
        coords={"t": t[:, 0], "marker": np.arange(n)},
    )


def test_orbit_poloidal_projection_and_quantities():
    orbits = orbits_dataset()
    result = orbits.plasma.plot.poloidal()
    labels = {line.get_label() for line in result.ax.lines}
    assert {"passing", "trapped", "lost"} <= labels
    lost = [line for line in result.ax.lines if line.get_label() == "lost"][0]
    assert len(lost.get_xdata()) == 150  # samples after the marker left are dropped
    coords, _, _ = hollow_torus(n1=6, n2=16, n3=4, cell_centred=True)
    with_boundary = plot_orbit_poloidal(
        orbits, boundary=scalar_field(coords, np.ones((6, 16, 4)))
    )
    assert "boundary" in {line.get_label() for line in with_boundary.ax.lines}

    quantities = plot_orbit_quantities(orbits, markers=4)
    assert len(quantities.ax) == 2
    mu_lines = quantities.ax[1].lines
    assert all(
        abs(line.get_ydata()[0]) < 1e-12 for line in mu_lines
    )  # drift of mu starts at zero
    classes = {line.get_label().split("(")[-1] for line in quantities.ax[0].lines}
    assert len(classes) >= 2  # markers spread over the classes
    assert (
        len(orbits.plasma.plot.quantities(quantities=("v_par",), markers=[0, 1]).ax)
        == 1
    )


def test_mode_numbers_scaled_to_the_full_torus_and_profiles_at_one_time():
    eta1, eta2, eta3 = np.linspace(0, 1, 9), np.arange(32) / 32, np.arange(8) / 8
    R, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    field = xr.DataArray(
        np.sin(np.pi * R) * np.cos(2 * np.pi * (10 * E2 - E3)),
        dims=("eta1", "eta2", "eta3"),
        coords={"eta1": eta1, "eta2": eta2, "eta3": eta3},
        name="u",
    )
    amplitudes = sp.mode_amplitudes(sp.mode_spectrum(field, scale=(1, 6)), top=1)
    assert amplitudes.mode.values.tolist() == ["(10, -6)"]
    profiles = field.plasma.plot.mode_profiles(top=1, scale=(1, 6))
    assert profiles.ax[0].get_legend().get_texts()[0].get_text() == "(m, n) = (10, -6)"
    assert len(profiles.ax) == 1  # real amplitudes: no phase panel
    with pytest.raises(ValueError, match="select a time"):
        field.expand_dims(t=[0.0, 1.0]).plasma.plot.mode_profiles()


def test_frequency_from_an_oscillating_energy_with_a_polynomial_detrend():
    t = np.linspace(0, 100, 501)
    energy = xr.DataArray(
        np.exp(0.01 * t) * (1 + np.cos(2 * 0.7 * t)) / 2, dims="t", coords={"t": t}
    )
    peak = sp.spectral_peaks(energy, n_peaks=1, detrend=2, window="hann")
    assert float(peak.omega_refined[0]) == pytest.approx(1.4, rel=5e-3)


def test_vtk_export_of_a_time_series(tmp_path):
    pv = pytest.importorskip("pyvista")
    coords, _, (X, _, _) = hollow_torus(n1=4, n2=8, n3=6)
    series = xr.DataArray(
        np.stack([X, 2 * X]),
        dims=("t", "eta1", "eta2", "eta3"),
        coords={"t": [0.0, 0.5], **coords},
        name="p",
    )
    paths = series.plasma.data.to_vtk(tmp_path / "frames")
    assert paths[0].endswith("p.pvd") and len(paths) == 3
    grid = pv.read(paths[2])
    np.testing.assert_allclose(grid["p"], 2 * X.ravel(order="F"))
    assert '<DataSet timestep="0.5"' in open(paths[0]).read()
    single = series.isel(t=0).plasma.data.to_vtk(tmp_path / "one.vts")
    assert pv.read(single[0]).n_points == X.size
