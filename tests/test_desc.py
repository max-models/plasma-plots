"""DESC equilibria evaluated into Datasets (from_desc), checked against DESC itself.

Uses DESC's own example equilibria (W7-X, and the axisymmetric DSHAPE), on small grids: the
values of from_desc must equal eq.compute at the same points, the volume from sqrt(g) DESC's V,
and the flux-surface average of |B| DESC's <|B|>.
"""

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("desc")
import desc.examples  # noqa: E402
from desc.grid import LinearGrid  # noqa: E402

import plasma_plots  # noqa: E402
from plasma_plots.analysis import (
    surface_average,  # noqa: E402
    volume_integral,
)
from plasma_plots.arrays import angle_period, logical_dims  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def w7x():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return desc.examples.get("W7-X")


@pytest.fixture(scope="module")
def ev(w7x):
    return plasma_plots.from_desc(
        w7x,
        ["|B|", "iota", "sqrt(g)", "B", "V", "p", "theta_PEST"],
        rho=5,
        theta=7,
        zeta=4,
    )


def point(eq, names, rho, theta, zeta):
    """DESC's own values at one point."""
    grid = LinearGrid(
        rho=np.array([rho]), theta=np.array([theta]), zeta=np.array([zeta]), NFP=eq.NFP
    )
    return eq.compute(names, grid=grid)


def test_the_layout(ev):
    assert ev["|B|"].dims == ("rho", "theta", "zeta")
    assert ev.B.dims == ("component", "rho", "theta", "zeta")
    assert list(ev.component.values) == ["x", "y", "z"]
    assert ev.iota.dims == ("rho",) and ev.V.dims == ()
    assert {"X", "Y", "Z", "theta_P"} <= set(ev.coords)
    np.testing.assert_allclose(ev.rho, np.linspace(0, 1, 5))
    np.testing.assert_allclose(
        ev.zeta, np.linspace(0, 2 * np.pi / 5, 4, endpoint=False)
    )
    assert ev.attrs["nfp"] == 5 and ev["|B|"].attrs["nfp"] == 5
    assert angle_period(ev["|B|"], "theta") == pytest.approx(2 * np.pi)
    assert angle_period(ev["|B|"], "zeta") == pytest.approx(2 * np.pi / 5)
    assert logical_dims(ev["|B|"]) == ("rho", "theta", "zeta")
    assert (
        ev["|B|"].attrs["label"] == r"$|\mathbf{B}|$"
        and ev["|B|"].attrs["units"] == r"$\mathrm{T}$"
    )
    assert "units" not in ev.iota.attrs  # dimensionless


def test_the_angle_coordinates_have_a_period(w7x, ev):
    # coordinate_lines draws contours of an angle across its seam by its period
    assert angle_period(ev["|B|"], "theta_P") == pytest.approx(2 * np.pi)
    pest = plasma_plots.from_desc(w7x, "|B|", rho=[1.0], theta=4, zeta=2, sfl="pest")
    assert angle_period(pest["|B|"], "theta") == pytest.approx(2 * np.pi)


@pytest.mark.parametrize("i, j, k", [(1, 2, 3), (4, 6, 0), (2, 0, 1)])
def test_values_match_desc_point_by_point(w7x, ev, i, j, k):
    rho, theta, zeta = (
        float(ev[d][n]) for d, n in zip(("rho", "theta", "zeta"), (i, j, k))
    )
    d = point(
        w7x, ["|B|", "B", "X", "Y", "Z", "phi", "theta_PEST", "iota"], rho, theta, zeta
    )
    assert ev["|B|"].values[i, j, k] == pytest.approx(d["|B|"][0])
    for axis in "XYZ":
        assert ev[axis].values[i, j, k] == pytest.approx(d[axis][0])
    assert ev.theta_P.values[i, j, k] == pytest.approx(d["theta_PEST"][0])
    assert ev.iota.values[i] == pytest.approx(d["iota"][0])
    (BR, Bphi, BZ), phi = d["B"][0], d["phi"][0]
    np.testing.assert_allclose(
        ev.B.values[:, i, j, k],
        [
            BR * np.cos(phi) - Bphi * np.sin(phi),
            BR * np.sin(phi) + Bphi * np.cos(phi),
            BZ,
        ],
    )


def test_cartesian_vectors_have_the_right_magnitude(ev):
    np.testing.assert_allclose(np.sqrt((ev.B**2).sum("component")), ev["|B|"])


def test_volume_and_surface_average_match_desc(w7x):
    fine = plasma_plots.from_desc(
        w7x, ["|B|", "sqrt(g)", "V"], rho=33, theta=32, zeta=24
    )
    volume = (
        float(volume_integral(fine["|B|"] * 0 + 1, jacobian=fine["sqrt(g)"])) * w7x.NFP
    )
    assert volume == pytest.approx(float(fine.V), rel=1e-3)
    surface = fine.sel(rho=[0.5])
    average = float(
        surface_average(surface["|B|"], jacobian=surface["sqrt(g)"]).squeeze()
    )
    grid = LinearGrid(rho=np.array([0.5]), M=16, N=12, NFP=w7x.NFP)
    assert average == pytest.approx(
        w7x.compute("<|B|>", grid=grid)["<|B|>"][0], rel=1e-6
    )
    assert (
        surface_average(surface["|B|"], jacobian=surface["sqrt(g)"]).attrs["units"]
        == r"$\mathrm{T}$"
    )


def test_a_pest_grid(w7x, ev):
    pest = plasma_plots.from_desc(
        w7x,
        ["|B|", "iota", "theta_PEST"],
        rho=[0.0, 0.5, 1.0],
        theta=8,
        zeta=4,
        sfl="pest",
    )
    assert pest["|B|"].dims == ("rho", "theta_P", "zeta")
    assert pest.theta_P.dims == (
        "theta_P",
    )  # "theta_PEST" among the names doesn't replace it
    assert pest.theta.dims == (
        "rho",
        "theta_P",
        "zeta",
    )  # DESC's own angle, a coordinate
    assert logical_dims(pest["|B|"]) == ("rho", "theta_P", "zeta")
    np.testing.assert_allclose(pest.iota, ev.iota.interp(rho=[0.0, 0.5, 1.0]))
    i, j, k = 1, 3, 2
    d = point(
        w7x, ["|B|", "theta_PEST"], 0.5, float(pest.theta[i, j, k]), float(pest.zeta[k])
    )
    assert pest["|B|"].values[i, j, k] == pytest.approx(d["|B|"][0])
    assert np.angle(
        np.exp(1j * (d["theta_PEST"][0] - float(pest.theta_P[j])))
    ) == pytest.approx(0, abs=1e-5)


def test_the_full_torus(w7x):
    torus = plasma_plots.from_desc(
        w7x,
        "|B|",
        rho=[1.0],
        theta=6,
        zeta=np.linspace(0, 2 * np.pi, 10, endpoint=False),
    )
    np.testing.assert_allclose(
        torus["|B|"].values[..., :2], torus["|B|"].values[..., 2:4]
    )  # nfp = 5


def test_an_axisymmetric_equilibrium():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = desc.examples.get("DSHAPE")
    ev = plasma_plots.from_desc(eq, ["|B|", "p"], rho=4, theta=6, zeta=3)
    assert ev.attrs["nfp"] == 1
    np.testing.assert_allclose(ev["|B|"].values[..., 0], ev["|B|"].values[..., 1])


def test_unsupported_names_are_refused(w7x):
    with pytest.raises(ValueError, match="no quantities"):
        plasma_plots.from_desc(w7x, ["not a quantity"])
    with pytest.raises(ValueError, match="shapes"):
        plasma_plots.from_desc(w7x, ["grad(B)"])
    with pytest.raises(ValueError, match="X, Y, Z"):
        plasma_plots.from_desc(w7x, ["x"])
    with pytest.raises(ValueError, match="sfl"):
        plasma_plots.from_desc(w7x, ["|B|"], sfl="boozer")


def test_the_plots_draw(ev):
    ev["|B|"].plasma.plot.slice(
        coords="physical",
        plane="RZ",
        zeta=0.0,
        overlays={"coordinate_lines": {"rho": 4, "theta_P": 8}},
    )
    ev.iota.plasma.plot.lineout(rationals=4)
    ev["|B|"].plasma.plot.slice(x="zeta", y="theta", rho=1.0)
    plt.close("all")
