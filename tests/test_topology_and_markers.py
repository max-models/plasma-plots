"""Boozer harmonics and quasi-symmetry, O- and X-points with the reconnected flux, and the marker
diagnostics (weights, sampling density, losses) of plasma_plots.analysis, with their plots.
"""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

import plasma_plots  # noqa: E402
from plasma_plots.analysis import critical_points  # noqa: E402
from plasma_plots.analysis import (
    boozer_spectrum,
    loss_map,
    lost_fraction,
    marker_density,
    quasisymmetry_error,
    reconnected_flux,
    weight_statistics,
)
from plasma_plots.plotting import View  # noqa: E402
from plasma_plots.plotting import (
    PlotResult,
    plot_critical_points,
    plot_loss_map,
    plot_lost_fraction,
    plot_marker_density,
    plot_weight_histogram,
)
from plasma_plots.spectral_plots import plot_boozer_spectrum  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# --- Boozer harmonics -------------------------------------------------------------------------


def boozer_field(nfp=3, breaking=0.02):
    """|B| = 1 + 0.1 ρ cos θ_B + breaking ρ² cos(2 θ_B − 3 ζ_B) on a Boozer grid: axisymmetric but
    for the one (2, −3) harmonic (in the cos(m θ + n ζ) convention)."""
    rho = np.linspace(0.2, 1, 5)
    theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / nfp, 24, endpoint=False)
    R, TH, ZE = np.meshgrid(rho, theta, zeta, indexing="ij")
    return xr.DataArray(
        1 + 0.1 * R * np.cos(TH) + breaking * R**2 * np.cos(2 * TH - 3 * ZE),
        dims=("rho", "theta_B", "zeta_B"),
        coords={
            "rho": rho,
            "theta_B": ("theta_B", theta, {"period": 2 * np.pi}),
            "zeta_B": ("zeta_B", zeta, {"period": 2 * np.pi / nfp}),
        },
        name="mod_B",
        attrs={"label": "$|B|$", "nfp": nfp},
    )


def test_boozer_spectrum_recovers_the_seeded_harmonics():
    B = boozer_field()
    B_mn = boozer_spectrum(B, top=3)
    assert B_mn.dims == ("mode", "rho") or B_mn.dims == ("rho", "mode")
    assert B_mn.name == "B_mn" and list(B_mn.mode.values) == [
        "(0, 0)",
        "(1, 0)",
        "(2, -3)",
    ]
    np.testing.assert_allclose(B_mn.sel(mode="(0, 0)"), 1.0, atol=1e-12)
    np.testing.assert_allclose(B_mn.sel(mode="(1, 0)"), 0.1 * B.rho, atol=1e-12)
    np.testing.assert_allclose(B_mn.sel(mode="(2, -3)"), 0.02 * B.rho**2, atol=1e-12)
    assert B_mn.attrs["nfp"] == 3
    assert B.plasma.analysis.boozer_spectrum(top=2).sizes["mode"] == 2


def test_quasisymmetry_error_measures_the_breaking_harmonic():
    B = boozer_field()
    qa = quasisymmetry_error(B, helicity="QA")
    assert qa.name == "quasisymmetry_error" and qa.dims == ("rho",)
    np.testing.assert_allclose(qa, 0.02 * B.rho**2, atol=1e-12)
    assert qa.attrs["helicity"] == [1, 0]
    # the (2, −3) harmonic has helicity (M, N) = (2, 3): nothing breaks that symmetry but (1, 0)
    np.testing.assert_allclose(quasisymmetry_error(B, helicity=(2, 3)), 0.1 * B.rho, atol=1e-12)
    # QH tries both handednesses of (1, nfp) and keeps the smaller error
    qh = quasisymmetry_error(B, helicity="QH")
    assert qh.attrs["helicity"][0] == 1 and abs(qh.attrs["helicity"][1]) == 3
    assert float(qh.isel(rho=-1)) == pytest.approx(np.hypot(0.1, 0.02))
    # quasi-poloidal: everything with m ≠ 0 breaks it
    np.testing.assert_allclose(
        quasisymmetry_error(B, helicity="QP"),
        np.hypot(0.1 * B.rho, 0.02 * B.rho**2),
        atol=1e-12,
    )
    assert float(quasisymmetry_error(boozer_field(breaking=0.0), helicity="QA").max()) < 1e-12
    with pytest.raises(ValueError, match="helicity"):
        quasisymmetry_error(B, helicity="QX")
    assert B.plasma.analysis.quasisymmetry_error().equals(qa)


def test_boozer_tools_want_boozer_angles_unless_told_otherwise():
    other = boozer_field().rename(theta_B="theta", zeta_B="zeta")
    with pytest.raises(ValueError, match="Boozer angles"):
        boozer_spectrum(other)
    assert boozer_spectrum(other, angles="any").sizes["mode"] > 0
    with pytest.raises(ValueError, match="angles"):
        boozer_spectrum(other, angles="pest")


def test_boozer_spectrum_plot_has_the_error_panel():
    result = plot_boozer_spectrum(boozer_field(), top=3, helicity="QA")
    assert isinstance(result, PlotResult) and len(result.ax) == 2
    legend = [t.get_text() for t in result.ax[0].get_legend().get_texts()]
    assert "(2, -3) breaking" in legend and "(1, 0)" in legend
    assert set(result.data) == {"amplitudes", "error"}
    single = plot_boozer_spectrum(boozer_field(), top=2)
    assert not isinstance(single.ax, (list, np.ndarray)) and set(single.data) == {"amplitudes"}
    assert isinstance(boozer_field().plasma.plot.boozer_spectrum(top=2), PlotResult)


# --- O- and X-points --------------------------------------------------------------------------


def flux_box(values, periodic=True, n=48):
    e = (np.arange(n) + 0.5) / n
    E1, E2 = np.meshgrid(e, e, indexing="ij")
    X, Y = 2 * np.pi * E1, 2 * np.pi * E2
    data = values(X, Y)
    dims = ("t", "eta1", "eta2", "eta3") if data.ndim == 3 else ("eta1", "eta2", "eta3")
    coords = {
        "eta1": ("eta1", e, {"period": 1.0} if periodic else {}),
        "eta2": ("eta2", e, {"period": 1.0} if periodic else {}),
        "eta3": [0.0],
        "X": (("eta1", "eta2", "eta3"), X[..., None]),
        "Y": (("eta1", "eta2", "eta3"), Y[..., None]),
        "Z": (("eta1", "eta2", "eta3"), 0 * X[..., None]),
    }
    if data.ndim == 3:
        coords["t"] = np.arange(data.shape[0], dtype=float)
    return xr.DataArray(data[..., None], dims=dims, coords=coords, name="A", attrs={"label": "$A$"})


def test_critical_points_of_a_magnetic_island():
    # an island chain: −cos(y − 0.3) + 0.3 cos(x − 0.7); O-points where both terms peak
    A = flux_box(lambda X, Y: -np.cos(Y - 0.3) + 0.3 * np.cos(X - 0.7))
    points = critical_points(A)
    frame = points.to_dataframe()
    assert list(frame.kind) == ["O", "O", "X", "X"]  # O first, each sorted by value
    o_points = frame[frame.kind == "O"]
    np.testing.assert_allclose(sorted(o_points.value), [-1.3, 1.3], atol=5e-3)
    maximum = o_points.iloc[-1]
    assert maximum.sign == 1 and (maximum.X, maximum.Y) == pytest.approx((0.7, np.pi + 0.3), abs=1e-3)
    x_points = frame[frame.kind == "X"]
    np.testing.assert_allclose(sorted(x_points.value), [-0.7, 0.7], atol=5e-3)
    assert (x_points.sign == 0).all()
    assert points.attrs["plane"] == ["eta1", "eta2"]
    # unrefined: the cell centres
    coarse = critical_points(A, refine=False)
    assert coarse.sizes["point"] == 4
    assert abs(float(coarse.value.max()) - 1.3) < 0.01


def test_a_bounded_box_loses_the_points_on_its_edges():
    A = flux_box(lambda X, Y: -np.cos(Y) + 0.3 * np.cos(X), periodic=False)
    points = critical_points(A)
    # only the saddle at (π, π) is interior; the extrema sit on the edges
    assert list(points.kind.values) == ["X"]
    assert points.X.item() == pytest.approx(np.pi, abs=1e-6)


def test_reconnected_flux_grows_with_the_island():
    amplitudes = 0.05 * np.exp(0.5 * np.arange(6))
    A = flux_box(lambda X, Y: np.stack([-np.cos(Y) + a * np.cos(X) for a in amplitudes]))
    flux = reconnected_flux(A, relative=False)
    assert flux.dims == ("t",) and flux.name == "reconnected_flux"
    np.testing.assert_allclose(flux, 2 * amplitudes, rtol=5e-3)  # A(O) − A(X) on the separatrix
    relative = reconnected_flux(A)
    assert float(relative[0]) == 0.0
    pinned = reconnected_flux(A, relative=False, o_point=(0.0, 0.5), x_point=(0.5, 0.5))
    np.testing.assert_allclose(pinned, 2 * amplitudes, rtol=5e-3)
    assert float(flux.plasma.analysis.growth_rate(window=(1.0, 5.0)).rate) == pytest.approx(0.5, rel=0.05)
    with pytest.raises(ValueError, match="missing dimensions"):
        reconnected_flux(A.isel(t=0))
    assert A.plasma.analysis.reconnected_flux(relative=False).equals(flux)
    assert A.plasma.analysis.critical_points().sizes["t"] == 6
    assert A.plasma.data.critical_points(t=-1).sizes["point"] == 4


def test_critical_points_need_a_plane():
    A = flux_box(lambda X, Y: -np.cos(Y) + 0.3 * np.cos(X))
    with pytest.raises(ValueError, match="two logical directions"):
        critical_points(A.isel(eta2=0))


def test_critical_points_plot_marks_both_kinds():
    A = flux_box(lambda X, Y: np.stack([-np.cos(Y) + a * np.cos(X) for a in (0.1, 0.3)]))
    result = plot_critical_points(A, view=View(isel={"t": -1, "eta3": 0}), label_values=True)
    assert isinstance(result, PlotResult) and "points" in result.data
    legend = [t.get_text() for t in result.ax.get_legend().get_texts()]
    assert legend == ["O-points (2)", "X-points (2)"]
    physical = plot_critical_points(A, view=View(isel={"t": -1, "eta3": 0}, coordinates="physical"))
    assert physical.ax.get_xlabel() == "X"
    with pytest.raises(ValueError, match="select one 't'"):
        plot_critical_points(A, view=View(isel={"eta3": 0}))
    assert isinstance(A.plasma.plot.critical_points(t=-1, eta3=0), PlotResult)
    assert isinstance(A.plasma.plot.critical_points(coords="physical", t=0, eta3=0), PlotResult)


# --- markers ----------------------------------------------------------------------------------


def markers(nt=20, nm=200, seed=0):
    """δf-like markers: weights spreading in time, a quarter of the markers lost at known times."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 10, nt)
    w0 = 1e-3 * rng.standard_normal(nm)
    w = w0[None] * np.exp(0.3 * t)[:, None]
    eta1 = np.sqrt(rng.uniform(0.04, 1, nm))
    v_par = rng.normal(0, 1, nm)
    mu = rng.exponential(0.5, nm)
    lost_from = np.where(np.arange(nm) < nm // 4, 5 + np.arange(nm) % 10, nt + 1)
    alive = np.arange(nt)[:, None] < lost_from[None]

    def column(values, **attrs):
        return (("t", "marker"), np.where(alive, values, 0.0), attrs)

    return (
        xr.Dataset(
            {
                "eta1": column(eta1[None] + 0 * w),
                "x": column(3 + 0 * w),
                "y": column(0 * w),
                "z": column(0.1 + 0 * w),
                "v_par": column(v_par[None] + 0 * w, label=r"$v_\parallel$"),
                "mu": column(mu[None] + 0 * w, label=r"$\mu$"),
                "weight": column(w, label="$w$"),
            },
            coords={"t": t, "marker": np.arange(nm)},
            attrs={"run": "synthetic markers"},
        ),
        alive,
    )


def test_weight_statistics_over_time():
    orbits, alive = markers()
    stats = weight_statistics(orbits)
    assert set(stats.data_vars) == {
        "mean",
        "std",
        "min",
        "max",
        "total",
        "noise",
        "effective_markers",
        "count",
    }
    np.testing.assert_allclose(stats["count"], alive.sum(axis=1))
    w = np.asarray(orbits.weight)[0]
    assert float(stats["std"][0]) == pytest.approx(w.std())
    assert float(stats["noise"][0]) == pytest.approx(np.sqrt((w**2).sum()) / abs(w.sum()))
    assert float(stats["effective_markers"][0]) == pytest.approx(w.sum() ** 2 / (w**2).sum())
    # lost markers drop out of the statistics
    last = np.asarray(orbits.weight)[-1][alive[-1]]
    assert float(stats["mean"][-1]) == pytest.approx(last.mean())
    with pytest.raises(ValueError, match="weights"):
        weight_statistics(orbits.drop_vars("weight"))
    assert orbits.plasma.analysis.weight_statistics().equals(stats)
    snapshot = weight_statistics(orbits.isel(t=0))
    assert snapshot["count"].ndim == 0


def test_marker_density_counts_and_weighs():
    orbits, alive = markers()
    sampling = marker_density(orbits, dims="eta1", bins=10)
    assert sampling.dims == ("t", "eta1") and sampling.name == "marker_density"
    np.testing.assert_allclose((sampling * 0.1).sum("eta1"), alive.sum(axis=1))  # per unit volume
    assert sampling.attrs["edges"][0][0] == 0.0 and sampling.attrs["edges"][0][-1] == 1.0
    weighted = marker_density(orbits, dims="eta1", bins=10, weight="weight")
    w = np.asarray(orbits.weight)
    np.testing.assert_allclose((weighted * 0.1).sum("eta1"), (w * alive).sum(axis=1), atol=1e-12)
    two = marker_density(orbits.isel(t=0), dims=("eta1", "v_par"), bins=(4, 6), ranges={"v_par": (-3, 3)})
    assert two.dims == ("eta1", "v_par") and two.shape == (4, 6)
    with pytest.raises(ValueError, match="not data variables"):
        marker_density(orbits, dims="r")
    assert orbits.plasma.analysis.marker_density(dims="eta1", bins=10).equals(sampling)


def test_lost_fraction_is_cumulative():
    orbits, alive = markers()
    fraction = lost_fraction(orbits)
    assert fraction.dims == ("t",) and float(fraction[0]) == 0.0
    assert float(fraction[-1]) == pytest.approx(0.25)
    assert (np.diff(fraction) >= 0).all()
    weighted = lost_fraction(orbits, weight="weight")
    w0 = np.abs(np.asarray(orbits.weight)[0])
    assert float(weighted[-1]) == pytest.approx(w0[:50].sum() / w0.sum())
    assert orbits.plasma.analysis.lost_fraction().equals(fraction)


def test_loss_map_gives_positions_and_loss_times():
    orbits, _ = markers()
    losses = loss_map(orbits)
    assert set(losses.data_vars) == {"v_par", "mu", "lost", "loss_time"}
    assert int(losses.lost.sum()) == 50
    t = np.asarray(orbits.t)
    np.testing.assert_allclose(losses.loss_time[:10], t[5 + np.arange(10) % 10])
    assert np.isnan(losses.loss_time[50:]).all()
    np.testing.assert_allclose(losses.v_par, np.asarray(orbits.v_par)[0])
    energy = loss_map(orbits, x="energy", y="pitch", absB=lambda x, y, z: 2 + 0 * x)
    v, mu = np.asarray(orbits.v_par)[0], np.asarray(orbits.mu)[0]
    np.testing.assert_allclose(energy.energy, 0.5 * v**2 + 2 * mu)
    with pytest.raises(ValueError, match="absB"):
        loss_map(orbits, x="energy")
    with pytest.raises(ValueError, match="neither"):
        loss_map(orbits, x="r")
    assert orbits.plasma.analysis.loss_map().equals(losses)
    assert orbits.plasma.data.loss_map().equals(losses)


def test_marker_plots():
    orbits, _ = markers()
    histogram = plot_weight_histogram(orbits, t=[0, 5.0, -1], bins=20)
    assert isinstance(histogram, PlotResult) and len(histogram.artists) == 3
    legend = [t.get_text() for t in histogram.ax.get_legend().get_texts()]
    assert legend[0].startswith("t = 0: mean") and legend[1].startswith("t = 5")
    assert histogram.data["statistics"].sizes["t"] == 3
    assert "relative noise of the total" in histogram.ax.get_title()
    reference = xr.DataArray(
        np.linspace(0.1, 1, 30),
        dims="eta1",
        coords={"eta1": np.linspace(0, 1, 30)},
        name="n",
    )
    density = plot_marker_density(orbits, x="eta1", against=reference, bins=10, t=-1)
    assert len(density.artists) == 3 and density.data["weighted"].name == "weighted_density"
    plain = plot_marker_density(orbits.drop_vars("weight"), x="eta1", bins=10)
    assert len(plain.artists) == 1 and plain.data["weighted"] is None
    lost = plot_lost_fraction(orbits)
    assert lost.ax.get_title().startswith("lost markers: 25%")
    by_weight = plot_lost_fraction(orbits, weight="weight", percent=False)
    assert by_weight.ax.get_title().startswith("lost particles")
    loss = plot_loss_map(orbits)
    assert len(loss.artists) == 2 and "losses" in loss.data
    legend = [t.get_text() for t in loss.ax.get_legend().get_texts()]
    assert legend[0].startswith("confined (150, 75%)") and legend[1].startswith("lost (50, 25%)")
    for method in ("weight_histogram", "marker_density", "lost_fraction", "loss_map"):
        assert isinstance(getattr(orbits.plasma.plot, method)(), PlotResult)
    with pytest.raises(ValueError, match="weights"):
        plot_weight_histogram(orbits.drop_vars("weight"))
