import numpy as np
import xarray as xr

from plasma_plots.arrays import repeat_periodic


def test_repeat_periodic_copies_torus_coordinates():
    R0, radius = 3.0, 0.5
    eta1 = np.array([0.5])
    eta2 = np.array([0.0, 0.25])
    eta3 = np.array([0.0, 0.5])
    theta = 2.0 * np.pi * eta2[None, :, None]
    phi = -2.0 * np.pi * eta3[None, None, :]
    radius_from_axis = R0 + radius * np.cos(theta)
    X = radius_from_axis * np.cos(phi) + 0.0 * eta1[:, None, None]
    Y = radius_from_axis * np.sin(phi) + 0.0 * eta1[:, None, None]
    Z = radius * np.sin(theta) + 0.0 * eta3[None, None, :]
    field = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("eta1", "eta2", "eta3"),
        coords={
            "eta1": eta1,
            "eta2": eta2,
            "eta3": eta3,
            "X": (("eta1", "eta2", "eta3"), X),
            "Y": (("eta1", "eta2", "eta3"), Y),
            "Z": (("eta1", "eta2", "eta3"), Z),
        },
    )

    full = repeat_periodic(field, poloidal_period=2, toroidal_period=3, R0=R0)

    assert full.sizes == {"eta1": 1, "eta2": 4, "eta3": 6}
    np.testing.assert_allclose(np.hypot(full.X, full.Y).max(), R0 + radius)
    np.testing.assert_allclose(full.Z.max(), radius)
    angle = -2.0 * np.pi / 3.0
    x0 = full.X.isel(eta2=0, eta3=0)
    y0 = full.Y.isel(eta2=0, eta3=0)
    np.testing.assert_allclose(
        full.X.isel(eta2=0, eta3=2), x0 * np.cos(angle) - y0 * np.sin(angle)
    )
    np.testing.assert_allclose(
        full.Y.isel(eta2=0, eta3=2), x0 * np.sin(angle) + y0 * np.cos(angle)
    )
    np.testing.assert_allclose(
        np.hypot(full.X.isel(eta2=2), full.Y.isel(eta2=2)),
        R0 - (np.hypot(full.X.isel(eta2=0), full.Y.isel(eta2=0)) - R0),
    )

    # Positive toroidal angle advances counter-clockwise
    phi_pos = 2.0 * np.pi * eta3[None, None, :]
    field_pos = field.assign_coords(
        X=(
            ("eta1", "eta2", "eta3"),
            radius_from_axis * np.cos(phi_pos) + 0.0 * eta1[:, None, None],
        ),
        Y=(
            ("eta1", "eta2", "eta3"),
            radius_from_axis * np.sin(phi_pos) + 0.0 * eta1[:, None, None],
        ),
    )
    full_pos = repeat_periodic(field_pos, poloidal_period=2, toroidal_period=3, R0=R0)
    pos_angle = 2.0 * np.pi / 3.0
    px0 = full_pos.X.isel(eta2=0, eta3=0)
    py0 = full_pos.Y.isel(eta2=0, eta3=0)
    np.testing.assert_allclose(
        full_pos.X.isel(eta2=0, eta3=2),
        px0 * np.cos(pos_angle) - py0 * np.sin(pos_angle),
    )
    np.testing.assert_allclose(
        full_pos.Y.isel(eta2=0, eta3=2),
        px0 * np.sin(pos_angle) + py0 * np.cos(pos_angle),
    )


def test_repeat_periodic_drops_closed_sector_endpoints():
    R0, radius = 3.0, 0.5
    eta = np.linspace(0.0, 1.0, 5)
    theta = 2.0 * np.pi * eta[:, None]
    phi = 2.0 * np.pi * eta[None, :]
    radius_from_axis = R0 + radius * np.cos(theta)
    X = radius_from_axis * np.cos(phi)
    Y = radius_from_axis * np.sin(phi)
    Z = radius * np.sin(theta) + 0.0 * phi
    field = xr.DataArray(
        np.zeros((5, 5)),
        dims=("eta2", "eta3"),
        coords={
            "eta2": eta,
            "eta3": eta,
            "X": (("eta2", "eta3"), X),
            "Y": (("eta2", "eta3"), Y),
            "Z": (("eta2", "eta3"), Z),
        },
    )

    repeated = repeat_periodic(field, poloidal_period=2, toroidal_period=3)

    assert repeated.sizes == {"eta2": 8, "eta3": 12}
