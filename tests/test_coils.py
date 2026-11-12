"""Coilsets and plasma boundaries (plasma_plots.coils) against closed forms and DESC itself.

Circular coils of radius a around a circular torus have curvature 1/a, length 2πa, a known
distance to the plasma and to each other; a circular torus has minor radius a. The expansion of
a stellarator-symmetric coilset must give DESC's coils in DESC's order.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

import plasma_plots
from plasma_plots.coils import coil_to_coil_distance, distance_to_surface

R0, A_COIL, A_PLASMA = 3.0, 1.0, 0.4


def circular_coilset(n_unique=3, nfp=2, sym=True, order=2):
    """Planar circular coils of radius A_COIL at toroidal angles inside the first half period."""
    half = np.pi / nfp
    phis = (np.arange(n_unique) + 0.5) * half / n_unique
    width = 2 * order + 1
    x_n, y_n, z_n = (np.zeros((n_unique, width)) for _ in range(3))
    const, cos1, sin1 = (
        order,
        order + 1,
        order - 1,
    )  # [sin(N s) .. sin(s), 1, cos(s) .. cos(N s)]
    for i, phi in enumerate(phis):
        x_n[i, const], x_n[i, cos1] = R0 * np.cos(phi), A_COIL * np.cos(phi)
        y_n[i, const], y_n[i, cos1] = R0 * np.sin(phi), A_COIL * np.sin(phi)
        z_n[i, sin1] = A_COIL
    return {
        "coil_x_n": x_n.tolist(),
        "coil_y_n": y_n.tolist(),
        "coil_z_n": z_n.tolist(),
        "currents": [1e6 * (i + 1) for i in range(n_unique)],
        "n_field_periods": nfp,
        "is_stellarator_symmetric": sym,
    }


def circular_boundary(nfp=2):
    r_cos = np.zeros((2, 3))
    z_sin = np.zeros((2, 3))
    r_cos[0, 1], r_cos[1, 1], z_sin[1, 1] = R0, A_PLASMA, A_PLASMA
    return {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "r_sin": None,
        "z_cos": None,
        "n_field_periods": nfp,
        "is_stellarator_symmetric": True,
    }


@pytest.mark.parametrize("sym", [True, False])
def test_expansion_counts_and_layout(sym):
    coils = plasma_plots.from_coilset(circular_coilset(sym=sym), n_points=32)
    assert coils.sizes == {"coil": 3 * 2 * (2 if sym else 1), "s": 32}
    assert coils.attrs["nfp"] == 2 and coils.attrs["fourier_order"] == 2
    assert int(coils.mirrored.sum()) == (6 if sym else 0)
    assert coils.X.attrs["units"] == "m" and coils.curvature.attrs["units"] == "1/m"
    # every copy carries the current of the coil it copies
    expected = 1e6 * (coils.unique_coil.values + 1)
    np.testing.assert_allclose(coils.current, expected)


def test_circles_have_exact_curvature_length_and_radius():
    coils = plasma_plots.from_coilset(circular_coilset(), n_points=64)
    np.testing.assert_allclose(coils.curvature, 1 / A_COIL, rtol=1e-12)
    np.testing.assert_allclose(coils.length, 2 * np.pi * A_COIL, rtol=1e-12)
    # every copy is a circle of radius A_COIL around a point at R0 in the midplane
    centers = coils[["X", "Y", "Z"]].reset_coords()[["X", "Y", "Z"]].mean("s")
    radius_of_center = np.hypot(centers.X, centers.Y)
    np.testing.assert_allclose(radius_of_center, R0, atol=1e-12)
    np.testing.assert_allclose(centers.Z, 0.0, atol=1e-12)


def test_copies_fill_the_torus_evenly():
    """3 unique coils per half period, mirrored and rotated: 12 coils, 30 degrees apart."""
    coils = plasma_plots.from_coilset(circular_coilset(), n_points=16)
    centers = coils.reset_coords()[["X", "Y"]].mean("s")
    phi = np.sort(np.mod(np.arctan2(centers.Y, centers.X), 2 * np.pi))
    np.testing.assert_allclose(np.diff(phi), np.pi / 6, atol=1e-12)


def test_inputs_json_and_attributes_agree():
    data = circular_coilset()

    class Coilset:
        pass

    obj = Coilset()
    for key, value in data.items():
        setattr(obj, key, value)
    a = plasma_plots.from_coilset(data, n_points=8)
    b = plasma_plots.from_coilset(json.dumps(data), n_points=8)
    c = plasma_plots.from_coilset(obj, n_points=8)
    xr.testing.assert_identical(a, b)
    xr.testing.assert_identical(a, c)


def test_bad_coilsets_raise():
    data = circular_coilset()
    with pytest.raises(ValueError, match="lacks"):
        plasma_plots.from_coilset({k: v for k, v in data.items() if k != "currents"})
    with pytest.raises(TypeError, match="lacks"):
        plasma_plots.from_coilset(object())
    with pytest.raises(ValueError, match="one current per coil"):
        plasma_plots.from_coilset({**data, "currents": [1.0]})
    with pytest.raises(ValueError, match="2N \\+ 1"):
        plasma_plots.from_coilset({**data, "coil_x_n": np.zeros((3, 4)).tolist()})
    with pytest.raises(ValueError, match="at least 3"):
        plasma_plots.from_coilset(data, n_points=2)


def test_boundary_circular_torus():
    plasma = plasma_plots.from_boundary(circular_boundary(), theta=48, zeta=None)
    assert plasma.R.dims == ("rho", "theta", "zeta")
    assert plasma.sizes["zeta"] == 24 * 2
    np.testing.assert_allclose(plasma.attrs["minor_radius"], A_PLASMA, rtol=1e-12)
    assert plasma.attrs["major_radius"] == R0
    radius = np.hypot(np.hypot(plasma.X, plasma.Y) - R0, plasma.Z)
    np.testing.assert_allclose(radius, A_PLASMA, atol=1e-12)
    one_period = plasma_plots.from_boundary(circular_boundary(), theta=8, zeta=5)
    assert one_period.zeta.max() < np.pi


def test_boundary_matches_the_fourier_convention():
    """R = sum r_cos cos(m theta - nfp n phi), Z = sum z_sin sin(...), n from -N to N."""
    rng = np.random.default_rng(3)
    nfp = 3
    r_cos = 0.05 * rng.normal(size=(3, 5))
    z_sin = 0.05 * rng.normal(size=(3, 5))
    r_cos[0, 2] = 2.0
    r_cos[1, 2] = z_sin[1, 2] = 0.3
    plasma = plasma_plots.from_boundary(
        {"r_cos": r_cos, "z_sin": z_sin, "n_field_periods": nfp},
        theta=[0.7],
        zeta=[0.2],
    )
    angle = np.arange(3)[:, None] * 0.7 - nfp * np.arange(-2, 3)[None, :] * 0.2
    R = np.sum(r_cos * np.cos(angle))
    Z = np.sum(z_sin * np.sin(angle))
    np.testing.assert_allclose(float(plasma.R.squeeze()), R)
    np.testing.assert_allclose(float(plasma.Z.squeeze()), Z)
    np.testing.assert_allclose(float(plasma.X.squeeze()), R * np.cos(0.2))


def test_boundary_non_symmetric_terms_and_errors():
    data = circular_boundary()
    data["z_cos"] = np.zeros((2, 3)).tolist()
    data["z_cos"][0][1] = 0.1  # shift the torus up
    plasma = plasma_plots.from_boundary(json.dumps(data), theta=16, zeta=4)
    np.testing.assert_allclose(plasma.Z.mean(), 0.1, atol=1e-12)
    with pytest.raises(ValueError, match="2N \\+ 1"):
        plasma_plots.from_boundary({**data, "r_cos": np.zeros((2, 4)).tolist()})
    with pytest.raises(ValueError, match="at least one"):
        plasma_plots.from_boundary(data, theta=0)


def test_distances_of_circles_around_a_torus():
    coils = plasma_plots.from_coilset(circular_coilset(), n_points=128)
    plasma = plasma_plots.from_boundary(circular_boundary(), theta=256, zeta=None)
    # concentric with the plasma in the coil planes: the gap is A_COIL - A_PLASMA everywhere,
    # sampled to within the surface's grid spacing
    d = distance_to_surface(coils, plasma)
    assert d.dims == ("coil", "s") and d.attrs["units"] == "m"
    np.testing.assert_allclose(d, A_COIL - A_PLASMA, rtol=0, atol=2e-2)
    assert float(d.min()) >= A_COIL - A_PLASMA - 1e-12
    # neighbouring coils are 30 degrees apart; inboard points are closest
    gap = coil_to_coil_distance(coils)
    inboard = 2 * (R0 - A_COIL) * np.sin(np.pi / 12)
    np.testing.assert_allclose(float(gap.min()), inboard, rtol=1e-3)
    with pytest.raises(ValueError, match="at least two"):
        coil_to_coil_distance(coils.isel(coil=[0]))
    with pytest.raises(ValueError, match="X, Y, Z"):
        distance_to_surface(coils, xr.Dataset())
    with pytest.raises(ValueError, match="coil, s"):
        distance_to_surface(plasma, plasma)


def test_expansion_matches_desc():
    pytest.importorskip("desc")
    from desc.coils import CoilSet, FourierXYZCoil
    from desc.grid import LinearGrid

    data = circular_coilset(n_unique=2, nfp=3)
    # perturb the circles so that a wrong mirror or ordering shows
    rng = np.random.default_rng(0)
    for key in ("coil_x_n", "coil_y_n", "coil_z_n"):
        data[key] = (np.asarray(data[key]) + 0.05 * rng.normal(size=(2, 5))).tolist()
    desc_coils = CoilSet(
        *[
            FourierXYZCoil(
                current=data["currents"][i],
                X_n=data["coil_x_n"][i],
                Y_n=data["coil_y_n"][i],
                Z_n=data["coil_z_n"][i],
            )
            for i in range(2)
        ],
        NFP=3,
        sym=True,
    )
    n_points = 24
    grid = LinearGrid(
        zeta=n_points
    )  # s = 0, 2π/n, ... (an integer grid means 2n + 1 points)
    expected = np.asarray(desc_coils._compute_position(grid=grid, basis="xyz"))
    for source in (data, desc_coils):
        coils = plasma_plots.from_coilset(source, n_points=n_points)
        points = np.stack(
            [coils[axis].transpose("coil", "s").values for axis in "XYZ"], -1
        )
        np.testing.assert_allclose(points, expected, atol=1e-12)
    curvature = desc_coils.coils[0].compute("curvature", grid=grid)["curvature"]
    coils = plasma_plots.from_coilset(desc_coils, n_points=n_points)
    np.testing.assert_allclose(coils.curvature.isel(coil=0), curvature, rtol=1e-10)


def test_desc_coilset_of_other_coils_raises():
    pytest.importorskip("desc")
    from desc.coils import CoilSet, FourierPlanarCoil

    with pytest.raises(TypeError, match="FourierXYZCoil"):
        plasma_plots.from_coilset(CoilSet(FourierPlanarCoil(), NFP=1))


def test_coil_scene_draws_tubes_and_surface():
    pv = pytest.importorskip("pyvista")
    pv.OFF_SCREEN = True
    from plasma_plots.pyvista_plots import coil_polylines, pyvista_coils

    coils = plasma_plots.from_coilset(circular_coilset(), n_points=32)
    plasma = plasma_plots.from_boundary(circular_boundary(), theta=16, zeta=None)
    lines = coil_polylines(coils, color_by="curvature")
    assert lines.n_lines == 12 and lines.n_points == 12 * 33
    np.testing.assert_allclose(lines.point_data["curvature"], 1 / A_COIL)
    coils["plasma_distance"] = distance_to_surface(coils, plasma)
    for kwargs in (
        {},
        {"color_by": "current", "surface": plasma, "surface_opacity": 0.5},
        {"color_by": "plasma_distance", "surface": plasma.R, "tube_radius": 0},
        {"surface": plasma.R, "surface_clim": (0.0, 2.0)},
    ):
        plotter = coils.plasma.plot.coils_3d(**kwargs)
        assert "coils" in plotter.actors
        assert ("surface" in plotter.actors) == ("surface" in kwargs)
        image = plotter.screenshot(return_img=True, window_size=(200, 150))
        assert image.std() > 0
        if "surface_clim" in kwargs:
            mapper = plotter.actors["surface"].mapper
            assert tuple(mapper.scalar_range) == kwargs["surface_clim"]
        plotter.close()
    with pytest.raises(ValueError, match="color_by must be one of"):
        pyvista_coils(coils, color_by="nfp")
    with pytest.raises(ValueError, match="coil, s"):
        coil_polylines(plasma)
    with pytest.raises(ValueError, match="no variable"):
        pyvista_coils(coils, surface=plasma.drop_vars("R"))


def test_boundary_from_desc_reproduces_the_surface_and_its_minor_radius():
    desc_examples = pytest.importorskip("desc.examples")
    import warnings

    from desc.grid import LinearGrid

    from plasma_plots.coils import boundary_from_desc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = desc_examples.get("W7-X")
    boundary = boundary_from_desc(eq)
    theta = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / eq.NFP, 5, endpoint=False)
    plasma = plasma_plots.from_boundary(boundary, theta=theta, zeta=zeta)
    grid = LinearGrid(rho=np.array([1.0]), theta=theta, zeta=zeta, NFP=eq.NFP)
    values = eq.compute(["R", "Z", "a"], grid=grid)
    np.testing.assert_allclose(
        plasma.R.values[0], np.asarray(values["R"]).reshape(5, 7).T, atol=1e-12
    )
    np.testing.assert_allclose(
        plasma.Z.values[0], np.asarray(values["Z"]).reshape(5, 7).T, atol=1e-12
    )
    np.testing.assert_allclose(
        plasma.attrs["minor_radius"], float(values["a"]), rtol=1e-8
    )
