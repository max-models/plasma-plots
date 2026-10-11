"""GVEC's evaluation Datasets: reading them (from_gvec), and the tools that go with flux coordinates.

The fixture mirrors what ``gvec.State.evaluate`` / ``evaluate_sfl`` return (checked against a real
GVEC 1.5 run): dimensions ``rad``/``pol``/``tor`` with indexed ``rho``/``theta``/``zeta``, vectors
along ``xyz``, the position ``pos``, LaTeX ``symbol`` attributes, ``N_FP``, ``theta_P`` over
``(pol, rad, tor)``, and Gauss weights whose toroidal one counts every field period. The geometry
is a rotating ellipse, whose Jacobian (``R a b rho``) and volume are known exactly.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

import plasma_plots  # noqa: E402
from plasma_plots.analysis import rational_surfaces  # noqa: E402
from plasma_plots.analysis import surface_average, volume_integral
from plasma_plots.arrays import angle_period, logical_dims  # noqa: E402
from plasma_plots.gvec import from_gvec, is_gvec  # noqa: E402

R0, A, B, NFP = 3.0, 1.0, 0.6, 3


def gvec_dataset(nrho=9, ntheta=24, nzeta=18, *, sfl=None, integration=False, index=True):
    """A Dataset laid out as GVEC's: logical angles, or Boozer (``sfl="boozer"``) or PEST ones."""
    if integration:
        nodes, gauss = np.polynomial.legendre.leggauss(nrho)
        rho, rad_weight = (nodes + 1) / 2, gauss / 2
    else:
        rho = np.linspace(0.0, 1.0, nrho)
    theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / NFP, nzeta, endpoint=False)
    r, t, z = np.meshgrid(rho, theta, zeta, indexing="ij")
    alpha = NFP * z  # the ellipse turns once per field period
    u, v = r * A * np.cos(t), r * B * np.sin(t)
    X1 = R0 + u * np.cos(alpha) - v * np.sin(alpha)
    X2 = u * np.sin(alpha) + v * np.cos(alpha)
    pos = np.stack([X1 * np.cos(z), X1 * np.sin(z), X2])
    names = {
        None: ("theta", "zeta"),
        "boozer": ("theta_B", "zeta_B"),
        "pest": ("theta_P", "zeta"),
    }[sfl]
    coords = {
        "rho": ("rad", rho),
        names[0]: ("pol", theta),
        names[1]: ("tor", zeta),
        "xyz": ("xyz", ["x", "y", "z"]),
    }
    if integration:
        coords.update(
            rad_weight=("rad", rad_weight),
            pol_weight=2 * np.pi / ntheta,
            tor_weight=2 * np.pi / nzeta,
        )
    grid = ("rad", "pol", "tor")
    ds = xr.Dataset(
        {
            "X1": (
                grid,
                X1,
                {"long_name": "first reference coordinate", "symbol": "X^1"},
            ),
            "X2": (
                grid,
                X2,
                {"long_name": "second reference coordinate", "symbol": "X^2"},
            ),
            "pos": (
                ("xyz", *grid),
                pos,
                {"long_name": "position vector", "symbol": r"\mathbf{x}"},
            ),
            "mod_B": (
                grid,
                R0 / X1,
                {
                    "long_name": "modulus of the magnetic field",
                    "symbol": r"\left|\mathbf{B}\right|",
                },
            ),
            "Jac": (
                grid,
                X1 * A * B * r,
                {"long_name": "Jacobian determinant", "symbol": r"\mathcal{J}"},
            ),
            "iota": (
                ("rad",),
                0.625 + 0.35 * rho**2,
                {"long_name": "rotational transform", "symbol": r"\iota"},
            ),
            "N_FP": (
                (),
                NFP,
                {"long_name": "number of field periods", "symbol": r"N_\text{FP}"},
            ),
        },
        coords=coords,
    )
    if sfl is None:
        ds["theta_P"] = (
            ("pol", "rad", "tor"),
            np.moveaxis(t + 0.2 * np.sin(t), 0, 1),
            {"long_name": "poloidal angle in PEST coordinates", "symbol": r"\theta_P"},
        )
    else:
        ds["theta"] = (
            grid,
            t,
            {"long_name": "Logical poloidal angle", "symbol": r"\theta"},
        )
    for name, symbol in zip(("rho", *names), (r"\rho", rf"\{names[0]}", rf"\{names[1]}")):
        ds[name].attrs.update(long_name=name, symbol=symbol)
        if index:
            ds = ds.set_xindex(name)
    return ds


# --- reading GVEC's layout --------------------------------------------------------------------


@pytest.mark.parametrize("index", [True, False], ids=["in memory", "from netCDF"])
def test_from_gvec_uses_the_flux_coordinates_as_dimensions(index):
    raw = gvec_dataset(index=index)
    before = {name: dict(var.attrs) for name, var in raw.variables.items()}
    ev = from_gvec(raw)
    assert {name: dict(var.attrs) for name, var in raw.variables.items()} == before  # input untouched
    assert dict(ev.sizes) == {"rho": 9, "theta": 24, "zeta": 18, "component": 3}
    assert is_gvec(raw) and not is_gvec(ev)
    assert from_gvec(ev).identical(ev)  # idempotent
    assert ev.mod_B.dims == ("rho", "theta", "zeta")
    assert {"X", "Y", "Z", "X1", "X2", "theta_P"} <= set(ev.mod_B.coords)
    assert "pos" not in ev and ev.theta_P.dims == ("theta", "rho", "zeta")
    np.testing.assert_allclose(ev.X.values, raw.pos.sel(xyz="x").values)
    assert ev.mod_B.attrs["label"] == r"$|\mathbf{B}|$"  # mathtext without \left/\right
    assert ev.N_FP.attrs["label"] == r"$N_\mathrm{FP}$"
    assert ev.rho.attrs["label"] == r"$\rho$"
    assert angle_period(ev, "theta") == pytest.approx(2 * np.pi)
    assert angle_period(ev, "zeta") == pytest.approx(2 * np.pi / NFP)
    assert angle_period(ev, "theta_P") == pytest.approx(2 * np.pi) and angle_period(ev, "rho") is None
    assert ev.attrs["nfp"] == NFP and ev.iota.attrs["nfp"] == NFP


def test_from_gvec_on_boozer_and_pest_grids_and_single_variables():
    boozer = from_gvec(gvec_dataset(sfl="boozer"))
    assert boozer.mod_B.dims == ("rho", "theta_B", "zeta_B")
    assert boozer.theta.dims == ("rho", "theta_B", "zeta_B") and "theta" in boozer.mod_B.coords
    assert logical_dims(boozer.mod_B) == ("rho", "theta_B", "zeta_B")
    pest = from_gvec(gvec_dataset(sfl="pest"))
    assert pest.mod_B.dims == ("rho", "theta_P", "zeta")
    assert logical_dims(pest.mod_B.isel(rho=0)) == ("rho", "theta_P", "zeta")
    single = from_gvec(gvec_dataset().mod_B)  # no N_FP: nfp from the grid over one field period
    assert single.dims == ("rho", "theta", "zeta") and single.name == "mod_B"
    assert single.attrs["nfp"] == NFP and angle_period(single, "zeta") == pytest.approx(2 * np.pi / NFP)
    assert from_gvec(gvec_dataset(nzeta=1).mod_B, nfp=5).attrs["nfp"] == 5


def test_from_gvec_keeps_gauss_weights_over_the_sampled_grid():
    ev = from_gvec(gvec_dataset(integration=True))
    assert ev.rho_weight.dims == ("rho",)
    assert float(ev.theta_weight) == pytest.approx(2 * np.pi / 24)
    assert float(ev.zeta_weight) == pytest.approx(2 * np.pi / 18 / NFP)  # one field period, not all


def test_the_accessors_read_gvec_data_by_themselves():
    raw = gvec_dataset()
    assert raw.mod_B.plasma.data.slice(x="zeta", y="theta", rho=1.0).dims == (
        "zeta",
        "theta",
    )
    assert isinstance(
        raw.mod_B.plasma.plot.slice(x="zeta", y="theta", rho=1.0),
        plasma_plots.plotting.PlotResult,
    )
    assert float(raw.plasma.analysis.surface_average("mod_B").sel(rho=0.5)) > 0
    plt.close("all")


def test_logical_dims_picks_the_triple_of_the_data():
    struphy = xr.DataArray(np.zeros((2, 3)), dims=("eta1", "eta2"))
    assert logical_dims(struphy) == ("eta1", "eta2", "eta3")
    assert logical_dims(xr.DataArray(np.zeros(3), dims="t")) == ("eta1", "eta2", "eta3")
    ev = from_gvec(gvec_dataset())
    assert logical_dims(ev.mod_B.isel(rho=-1)) == ("rho", "theta", "zeta")


# --- spectra, integrals, averages -----------------------------------------------------------


def test_mode_spectrum_on_gvec_angles_gives_full_torus_mode_numbers():
    ev = from_gvec(gvec_dataset(ntheta=16, nzeta=12))
    field = np.cos(2 * ev.theta - 2 * NFP * ev.zeta) * xr.ones_like(ev.rho) + 0 * ev.mod_B
    field.attrs = ev.mod_B.attrs
    modes = field.plasma.analysis.mode_spectrum()
    assert set(np.asarray(modes.n) % NFP) == {0}
    amplitude = abs(modes).isel(rho=-1)
    assert float(amplitude.sel(m=2, n=-2 * NFP)) == pytest.approx(0.5)
    assert (
        float(
            amplitude.where((amplitude.m != 2) | (amplitude.n != -2 * NFP))
            .where((amplitude.m != -2) | (amplitude.n != 2 * NFP))
            .max()
        )
        < 1e-12
    )
    result = field.plasma.plot.mode_map(rho=-1)
    assert result.data["amplitude"].sizes["n"] == 12
    plt.close("all")


def test_volume_integral_over_one_field_period():
    exact = 2 * np.pi**2 * R0 * A * B / NFP  # Pappus: the ellipse's area times its path
    gauss = from_gvec(gvec_dataset(integration=True))
    assert float(volume_integral(xr.ones_like(gauss.mod_B), jacobian=gauss.Jac)) == pytest.approx(exact, rel=1e-12)
    uniform = from_gvec(gvec_dataset(nrho=21))
    numerical = float(volume_integral(xr.ones_like(uniform.mod_B)))  # √g from X, Y, Z
    assert numerical == pytest.approx(exact, rel=1e-2)


def test_surface_average_with_gvec_jacobian_and_numerical_one():
    ev = from_gvec(gvec_dataset(nrho=11))
    ones = surface_average(xr.ones_like(ev.mod_B), jacobian=ev.Jac)
    np.testing.assert_allclose(ones, 1.0)  # the axis too, where √g = 0: the plain mean there
    # <R0/R> with √g ∝ R: ∫ R0 dθ dζ / ∫ R dθ dζ = R0 / R0 = 1 on every surface
    exact = ev.plasma.analysis.surface_average("mod_B")
    assert exact.dims == ("rho",) and exact.attrs["label"].startswith("⟨")
    np.testing.assert_allclose(exact.isel(rho=slice(1, None)), 1.0, rtol=1e-12)
    numerical = ev.mod_B.plasma.analysis.surface_average()
    np.testing.assert_allclose(numerical.isel(rho=slice(1, None)), 1.0, rtol=1e-2)
    with pytest.raises(ValueError, match="angles"):
        surface_average(ev.mod_B.isel(zeta=0, drop=True), jacobian=ev.Jac.isel(zeta=0, drop=True))


def test_rational_surfaces_of_iota_and_q_profiles():
    rho = np.linspace(0, 1, 101)
    iota = xr.DataArray(0.625 + 0.35 * rho**2, dims="rho", coords={"rho": rho}, attrs={"nfp": 3})
    surfaces = rational_surfaces(iota, count=4)
    assert [(int(n), int(m)) for n, m in zip(surfaces.n, surfaces.m)] == [
        (3, 4),
        (6, 7),
        (9, 10),
        (9, 11),
    ]
    assert float(surfaces[0]) == pytest.approx(np.sqrt((0.75 - 0.625) / 0.35), abs=1e-3)
    q = xr.DataArray(1 + 2 * rho**2, dims="rho", coords={"rho": rho})
    assert [(int(n), int(m)) for n, m in zip(*(rational_surfaces(q).n, rational_surfaces(q).m))] == [
        (1, 1),
        (2, 1),
        (3, 1),
        (3, 2),
    ]
    hollow = xr.DataArray(0.45 - (rho - 0.5) ** 2, dims="rho", coords={"rho": rho})  # in [0.2, 0.45]
    twice = rational_surfaces(hollow, count=1, max_denominator=3)
    assert float(twice.value[0]) == pytest.approx(1 / 3) and twice.sizes["surface"] == 2  # both sides
    assert rational_surfaces(xr.DataArray(np.full(5, 0.55), dims="rho"), max_denominator=1).sizes["surface"] == 0
    with pytest.raises(ValueError, match="1-D"):
        rational_surfaces(xr.DataArray(np.zeros((2, 2)), dims=("a", "b")))


def test_lineout_marks_rational_surfaces():
    ev = from_gvec(gvec_dataset(nrho=21))
    result = ev.iota.plasma.plot.lineout(rationals=3)
    labels = [text.get_text().strip() for text in result.ax.texts]
    assert labels == ["3/4", "6/7", "9/10"]
    assert ev.iota.plasma.analysis.rational_surfaces(count=2).sizes["surface"] == 2
    plt.close("all")


# --- poloidal planes ---------------------------------------------------------------------------


def test_coordinate_lines_of_drawn_dims_and_of_an_angle_coordinate():
    ev = from_gvec(gvec_dataset())
    plain = ev.mod_B.plasma.plot.slice(coords="physical", plane="RZ", zeta=0.0)
    lines = ev.mod_B.plasma.plot.slice(
        coords="physical",
        plane="RZ",
        zeta=0.0,
        overlays={"coordinate_lines": {"rho": 4, "theta": 8}},
    )
    assert len(lines.ax.lines) - len(plain.ax.lines) == 4 + 8
    ring = lines.ax.lines[len(plain.ax.lines)]  # rho = 0.25: an ellipse of semi-axes a/4, b/4
    np.testing.assert_allclose(np.ptp(ring.get_xdata()), 2 * A / 4, rtol=1e-6)
    pest = ev.mod_B.plasma.plot.slice(
        coords="physical",
        plane="RZ",
        zeta=0.0,
        overlays={"coordinate_lines": {"theta_P": 6}},
    )
    contours = [c for c in pest.ax.collections if type(c).__name__ == "QuadContourSet"]
    assert len(contours) == 6
    assert all(len(c.allsegs[0]) >= 1 for c in contours)  # one line each, none at the seam twice
    with pytest.raises(ValueError, match="not a coordinate"):
        ev.mod_B.plasma.plot.slice(
            coords="physical",
            plane="RZ",
            zeta=0.0,
            overlays={"coordinate_lines": {"psi": 3}},
        )
    plt.close("all")


def test_x1x2_plane_and_panels_over_zeta_with_their_own_limits():
    ev = from_gvec(gvec_dataset())
    result = ev.mod_B.plasma.plot.slice(coords="physical", plane="X1X2", zeta=0.0)
    assert result.ax.get_xlabel() == "$X^1$"
    with pytest.raises(ValueError, match="X1"):
        ev.mod_B.drop_vars("X1").plasma.plot.slice(coords="physical", plane="X1X2", zeta=0.0)
    panels = ev.mod_B.plasma.plot.panels(
        sweep="zeta",
        coords="physical",
        plane="RZ",
        nrows=1,
        ncols=3,
        overlays={"coordinate_lines": {"rho": 2}},
    )
    tallest = float(ev.Z.isel(zeta=8).max())  # the middle panel: the ellipse turned by 160 degrees
    assert tallest > float(ev.Z.isel(zeta=0).max()) + 0.05
    assert panels.ax[0, 1].get_ylim()[1] >= tallest  # not clipped to the first panel's limits
    plt.close("all")


def test_new_options_convert_to_plotly_completely():
    pytest.importorskip("plotly")
    ev = from_gvec(gvec_dataset())
    with warnings.catch_warnings():
        warnings.simplefilter("error", plasma_plots.plotly_backend.ConversionWarning)
        ev.mod_B.plasma.plot.slice(
            coords="physical",
            plane="RZ",
            zeta=0.0,
            backend="plotly",
            overlays={"coordinate_lines": {"rho": 3, "theta_P": 4}},
        )
        figure = ev.iota.plasma.plot.lineout(rationals=2, backend="plotly").fig
    assert [a.text.strip() for a in figure.layout.annotations] == ["3/4", "6/7"]
    plt.close("all")


def test_pyvista_views_of_gvec_data():
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    ev = from_gvec(gvec_dataset())
    pieces = ev.mod_B.plasma.data.slices_3d(cuts={"rho": [1.0], "zeta": 0.0})
    assert [piece.shape for piece in pieces] == [(1, 24, 18), (9, 24, 1)]
    from plasma_plots.pyvista_plots import _vtk_text

    def domain_faces(cuts):
        plotter = ev.mod_B.plasma.plot.slices_3d(cuts=cuts)
        names = [name for name in plotter.actors if name.startswith("domain")]
        plotter.close()
        return len(names)

    # a cut on the domain's edge replaces that translucent context face (they'd flicker together)
    assert domain_faces({"rho": [1.0]}) == domain_faces({"rho": [0.5]}) - 1
    assert _vtk_text(r"$|\mathbf{B}|$ [T] |x|") == r"$\vert \mathbf{B}\vert $ [T] |x|"
