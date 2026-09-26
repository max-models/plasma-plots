"""Helpers and plots added after surveying all struphy-hub examples: errors against exact
solutions, mode projections, vector calculus, local components, orbit invariants, branch
tracing, reference overlays, line animations, measured-vs-theory plots, orbit grids and slice
overlays."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

import struphy_plots  # noqa: E402, F401
from struphy_plots import analysis as an  # noqa: E402
from struphy_plots.analysis import power_spectrum  # noqa: E402
from struphy_plots.plotting import (  # noqa: E402
    animate_lines,
    plot_lineout,
    plot_measured_vs_theory,
    plot_orbit_grid,
    plot_profiles,
    plot_timeseries,
)
from struphy_plots.spectral import trace_branch  # noqa: E402

R0 = 3.0


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def torus(n1=12, n2=33, n3=33):
    e1, e2, e3 = np.linspace(0, 1, n1), np.linspace(0, 1, n2), np.linspace(0, 1, n3)
    E1, E2, E3 = np.meshgrid(e1, e2, e3, indexing="ij")
    r, th, ph = 0.2 + 0.8 * E1, 2 * np.pi * E2, 2 * np.pi * E3
    X, Y, Z = (R0 + r * np.cos(th)) * np.cos(ph), (R0 + r * np.cos(th)) * np.sin(ph), r * np.sin(th)
    coords = {"eta1": e1, "eta2": e2, "eta3": e3, **{n: (("eta1", "eta2", "eta3"), c) for n, c in zip("XYZ", (X, Y, Z))}}
    return coords, (X, Y, Z), (th, ph)


def vector(coords, values, name="v"):
    return xr.DataArray(values, dims=("component", "eta1", "eta2", "eta3"), coords={"component": [0, 1, 2], **coords}, name=name)


def box(n=48):
    e = (np.arange(n) + 0.5) / n
    E1, E2 = np.meshgrid(e, e, indexing="ij")
    X, Y = 2 * np.pi * E1, 2 * np.pi * E2
    coords = {"eta1": e, "eta2": e, "eta3": [0.0], "X": (("eta1", "eta2", "eta3"), X[..., None]),
              "Y": (("eta1", "eta2", "eta3"), Y[..., None]), "Z": (("eta1", "eta2", "eta3"), 0 * X[..., None])}
    return coords, X, Y


def test_error_norms_against_an_exact_function():
    coords, (X, _, _), _ = torus()
    field = xr.DataArray(np.sin(X) + 0.01, dims=("eta1", "eta2", "eta3"), coords=coords, name="f")
    exact = lambda x, y, z: np.sin(x)  # noqa: E731
    assert float(an.error(field, exact)) == pytest.approx(0.01)
    assert float(an.error(field, exact, norm="max")) == pytest.approx(0.01)
    assert float(an.error(field, exact, norm="l1")) == pytest.approx(0.01)
    volume = 2 * np.pi**2 * R0 * (1 - 0.04)
    assert float(an.error(field, exact, norm="l2", weighted=True)) == pytest.approx(0.01 * np.sqrt(volume), rel=1e-6)
    assert float(an.error(field, exact, norm="rms", weighted=True)) == pytest.approx(0.01, rel=1e-6)
    pointwise = field.struphy.analysis.error(exact, norm="pointwise")
    np.testing.assert_allclose(pointwise, 0.01)
    series = field.expand_dims(t=[0.0, 1.0]).copy()
    per_time = an.error(series, lambda x, y, z, t: np.sin(x) + 0 * t)
    assert per_time.dims == ("t",)
    relative = an.error(field, np.sin(X) * 0 + 2.0, relative=True)
    assert float(relative) == pytest.approx(float(an.error(field, np.sin(X) * 0 + 2.0)) / 2.0)
    with pytest.raises(ValueError, match="norm"):
        an.error(field, exact, norm="l3")


def test_project_mode_with_bin_correction_and_phases():
    x = (np.arange(64) + 0.5) / 64
    signal = xr.DataArray(0.5 * np.sin(2 * np.pi * 3 * x) + 0.2 * np.cos(2 * np.pi * 5 * x), dims="eta1", coords={"eta1": x})
    assert float(an.project_mode(signal, dim="eta1", number=3)) == pytest.approx(0.5)
    assert float(an.project_mode(signal, dim="eta1", number=5, kind="cos")) == pytest.approx(0.2)
    complex_amplitude = complex(signal.struphy.analysis.project_mode(dim="eta1", number=3, kind="complex"))
    assert abs(complex_amplitude) == pytest.approx(0.5)
    assert np.angle(complex_amplitude) == pytest.approx(-np.pi / 2)
    edges = np.arange(65) / 64
    binned = xr.DataArray(
        0.5 * (np.cos(2 * np.pi * 3 * edges[:-1]) - np.cos(2 * np.pi * 3 * edges[1:])) / (2 * np.pi * 3 / 64),
        dims="eta1", coords={"eta1": x},
    )
    assert float(an.project_mode(binned, dim="eta1", number=3)) < 0.499
    assert float(an.project_mode(binned, dim="eta1", number=3, bin_correction=True)) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="full period"):
        an.project_mode(signal.isel(eta1=slice(0, 32)), dim="eta1", number=3)


def test_divergence_curl_and_local_components_on_a_torus():
    coords, (X, Y, _), (th, ph) = torus()
    rotation = vector(coords, np.stack([-Y, X, 0 * X]))
    np.testing.assert_allclose(rotation.struphy.analysis.divergence(), 0.0, atol=1e-10)
    np.testing.assert_allclose(an.curl(rotation).sel(component=2), 2.0, atol=1e-10)
    np.testing.assert_allclose(an.curl(rotation).sel(component=[0, 1]), 0.0, atol=1e-10)
    e_theta = vector(coords, np.stack([-np.sin(th) * np.cos(ph), -np.sin(th) * np.sin(ph), np.cos(th)]))
    local = e_theta.struphy.analysis.toroidal_components(R0=R0)
    assert list(local.component.values) == ["radial", "poloidal", "toroidal"]
    np.testing.assert_allclose(local.sel(component="poloidal"), 1.0, atol=1e-12)
    np.testing.assert_allclose(local.sel(component=["radial", "toroidal"]), 0.0, atol=1e-12)
    cylinder = an.cylindrical_components(rotation)
    np.testing.assert_allclose(cylinder.sel(component="phi"), np.hypot(X, Y), atol=1e-12)
    np.testing.assert_allclose(cylinder.sel(component="R"), 0.0, atol=1e-12)
    polar = an.polar_coordinates(rotation.isel(component=0))
    np.testing.assert_allclose(polar.r, np.hypot(X, Y))


def test_flux_function_recovers_a_known_flux():
    coords, X, Y = box()
    A = np.sin(X) * np.cos(Y)
    field = vector(coords, np.stack([-np.sin(X) * np.sin(Y), -np.cos(X) * np.cos(Y), 0 * X])[..., None], name="B")
    flux = field.struphy.analysis.flux_function()
    assert np.abs(np.asarray(flux) - (A - A.mean())).max() < 2e-3
    with pytest.raises(ValueError, match="Cartesian"):
        coords_t, (Xt, Yt, _), _ = torus(n3=1)
        an.flux_function(vector(coords_t, np.zeros((3, 12, 33, 1))))


def test_orbit_invariants_and_bounce_period():
    t = np.linspace(0, 40, 801)
    v_par = np.stack([np.cos(0.5 * t), 1.0 + 0 * t], axis=1)
    orbits = xr.Dataset(
        {"x": (("t", "marker"), np.full((801, 2), 3.0)), "y": (("t", "marker"), np.zeros((801, 2))),
         "z": (("t", "marker"), np.zeros((801, 2))), "v_par": (("t", "marker"), v_par),
         "mu": (("t", "marker"), 0.5 * (1 - v_par**2) / 2.0)},
        coords={"t": t, "marker": [0, 1]},
    )
    invariants = orbits.struphy.analysis.orbit_invariants(absB=lambda x, y, z: 2.0 + 0 * x)
    np.testing.assert_allclose(invariants.energy, 0.5, atol=1e-12)
    periods = orbits.struphy.analysis.bounce_period()
    assert float(periods[0]) == pytest.approx(2 * np.pi / 0.5, rel=1e-4)
    assert np.isnan(periods[1])  # passing
    full = xr.Dataset({n: (("t", "marker"), np.full((5, 1), v)) for n, v in (("v1", 3.0), ("v2", 4.0), ("v3", 0.0))},
                      coords={"t": np.arange(5.0), "marker": [0]})
    np.testing.assert_allclose(an.orbit_invariants(full).speed, 5.0)
    with pytest.raises(ValueError, match="no invariant"):
        an.orbit_invariants(orbits.drop_vars("mu"))


def test_trace_branch_follows_a_curved_branch_in_both_directions():
    t = np.linspace(0, 60, 400)
    x = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    X, T = np.meshgrid(x, t)
    omega = lambda k: np.sqrt(1 + 3 * k**2)  # noqa: E731
    right = sum(np.cos(k * X - omega(k) * T) for k in (1, 2, 3, 4, 5))
    for signal in (right, sum(np.cos(k * X + omega(k) * T) for k in (1, 2, 3, 4, 5))):
        field = xr.DataArray(signal, dims=("t", "eta1"), coords={"t": t, "eta1": x})
        traced = power_spectrum(field).struphy.analysis.trace_branch(omega, window=0.2, k_range=(0.5, 5.5)).dropna("k")
        assert traced.k.values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert float(abs(traced.relative_error).max()) < 0.01
        # without a k range, wavenumbers that carry no wave are left out
        everywhere = trace_branch(power_spectrum(field), omega).dropna("k")
        assert everywhere.k.values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    result = field.struphy.plot.dispersion(branches={"Bohm-Gross": omega}, frequencies={"cutoff": 1.0}, points={"traced": traced})
    labels = [text.get_text() for text in result.ax.get_legend().get_texts()]
    assert {"Bohm-Gross", "cutoff", "traced"} <= set(labels)


def test_reference_overlays_on_timeseries_lineouts_and_profiles():
    t = np.linspace(0, 5, 51)
    decay = xr.DataArray(np.exp(-0.5 * t), dims="t", coords={"t": t}, name="u")
    result = plot_timeseries(decay, reference={"+envelope": lambda t: np.exp(-0.5 * t), "-envelope": lambda t: -np.exp(-0.5 * t)}, logy=False)
    labels = [text.get_text() for text in result.ax.get_legend().get_texts()]
    assert "+envelope" in labels and "-envelope" in labels
    assert decay.struphy.plot.timeseries(reference=decay * 2, logy=False).ax.get_legend() is not None

    x = np.linspace(0, 1, 21)
    profile = xr.DataArray(np.sin(np.pi * x), dims="eta1", coords={"eta1": x, "t": 0.5}, name="p")
    line = plot_lineout(profile, x_of=lambda e: 2 * e, reference=lambda x, t: np.sin(np.pi * x / 2) + t * 0)
    np.testing.assert_allclose(line.artists[0].get_xdata()[[0, -1]], [0, 2])
    exact = line.artists[1]
    np.testing.assert_allclose(exact.get_ydata(), np.sin(np.pi * exact.get_xdata() / 2), atol=1e-12)

    times = np.linspace(0, 1, 5)
    moving = xr.DataArray(np.sin(np.pi * (x[None] - 0.1 * times[:, None])), dims=("t", "eta1"), coords={"t": times, "eta1": x})
    profiles = plot_profiles(moving, x="eta1", reference=lambda x, t: np.sin(np.pi * (x - 0.1 * t)))
    assert len(profiles.artists) == 8  # 4 profiles, 4 exact curves
    assert moving.struphy.plot.profiles(x="eta1", at=[0, 1], reference={"exact": lambda x, t: np.sin(np.pi * x)}).ax is not None


def test_line_animation_with_the_exact_profile():
    x = np.linspace(0, 1, 41)
    t = np.linspace(0, 2, 11)
    wave = xr.DataArray(np.sin(2 * np.pi * (x[None] - 0.3 * t[:, None])), dims=("t", "eta1"), coords={"t": t, "eta1": x}, name="phi")
    animation = wave.struphy.plot.line_animation(reference=lambda x, t: np.sin(2 * np.pi * (x - 0.3 * t)))
    assert isinstance(animation, FuncAnimation)
    animation._func(5)
    ax = animation._fig.axes[0]
    data, exact = ax.lines[0], ax.lines[1]
    np.testing.assert_allclose(data.get_ydata(), wave.isel(t=5))
    np.testing.assert_allclose(exact.get_ydata(), np.sin(2 * np.pi * (exact.get_xdata() - 0.3 * t[5])), atol=1e-12)
    lo, hi = ax.get_ylim()
    assert lo < -1 and hi > 1
    with pytest.raises(ValueError, match="select every dimension"):
        animate_lines(wave.expand_dims(eta2=[0.0, 1.0]))


def test_measured_against_theory_with_relative_errors():
    k = np.array([0.5, 1.0, 1.5, 2.0])
    theory = lambda k: np.sqrt(1 + 3 * k**2)  # noqa: E731
    measured = xr.DataArray(theory(k) * 1.01, dims="k", coords={"k": k}, name="omega")
    result = measured.struphy.plot.against_theory(theory)
    assert len(result.ax) == 2
    errors = result.ax[1].lines[0].get_ydata()
    np.testing.assert_allclose(errors, 0.01)
    several = plot_measured_vs_theory({"run A": (k, theory(k)), "run B": measured}, {"kinetic": theory, "fluid": lambda k: 1 + k}, logx=True)
    labels = [text.get_text() for text in several.ax[0].get_legend().get_texts()]
    assert {"run A", "run B", "kinetic", "fluid"} <= set(labels)
    assert plot_measured_vs_theory(measured, show_error=False).ax.get_title() == "Measured against theory"


def test_orbit_grid_one_panel_per_marker():
    t = np.linspace(0, 20, 100)[:, None]
    theta = np.linspace(0, 2 * np.pi, 6)[None] + 0.3 * t
    v_par = np.where(np.arange(6)[None] % 2 == 0, np.cos(0.4 * t), 1.0 + 0 * t)
    orbits = xr.Dataset(
        {"x": (("t", "marker"), (R0 + 0.5 * np.cos(theta)) * np.ones_like(t)), "y": (("t", "marker"), np.zeros((100, 6))),
         "z": (("t", "marker"), 0.5 * np.sin(theta)), "v_par": (("t", "marker"), v_par)},
        coords={"t": t[:, 0], "marker": np.arange(6)},
    )
    result = orbits.struphy.plot.orbit_grid(markers=4, ncols=2)
    assert result.ax.shape == (2, 2)
    titles = [ax.get_title() for ax in result.ax.ravel()]
    assert any("trapped" in title for title in titles) and any("passing" in title for title in titles)
    assert plot_orbit_grid(orbits, markers=[0, 1, 2], ncols=2).ax.ravel()[-1].get_visible() is False


def test_slice_overlays_second_field_boundary_grid_lines_and_points():
    coords, X, Y = box(24)
    A = xr.DataArray((np.sin(X) * np.cos(Y))[..., None], dims=("eta1", "eta2", "eta3"), coords=coords, name="A")
    J = A.copy(data=(2 * np.sin(X) * np.cos(Y))[..., None]).rename("J")
    result = J.struphy.plot.slice(
        coords="physical", plane="XY", eta3=0,
        overlays={"contours_of": A, "contour_levels": 6, "boundary": True, "grid_lines": 6,
                  "points": {"O-point": (np.pi / 2, 0.0)}, "lines": {"diagonal": lambda x: x}},
    )
    kinds = [type(c).__name__ for c in result.ax.collections]
    assert "QuadContourSet" in kinds
    labels = [text.get_text() for text in result.ax.get_legend().get_texts()]
    assert {"O-point", "diagonal"} <= set(labels)
    assert len(result.ax.lines) >= 4 + 1  # boundary edges, grid lines, the diagonal
    series = J.expand_dims(t=[0.0, 1.0]).copy()
    animation = series.struphy.plot.animation(coords="physical", plane="XY", eta3=0, overlays={"contours_of": A.expand_dims(t=[0.0, 1.0])})
    animation._func(1)
    ax = animation._fig.axes[0]
    assert sum(type(c).__name__ == "QuadContourSet" for c in ax.collections) == 1
    with pytest.raises(ValueError, match="unknown overlays"):
        J.struphy.plot.slice(coords="physical", plane="XY", eta3=0, overlays={"contour": A})
