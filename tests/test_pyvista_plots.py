"""Tests for the PyVista 3-D views, for 3-D fields and for 2-D ones (one flat direction)."""

import numpy as np
import pytest
import xarray as xr

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

import struphy_plots  # noqa: E402, F401  (registers the accessors)
from struphy_plots import pyvista_plots as p3  # noqa: E402


def torus_coords(n1=6, n2=12, n3=16, R0=3.0, full=True):
    eta1, eta2, eta3 = np.linspace(0, 1, n1), np.linspace(0, 1, n2), np.linspace(0, 1, n3)
    E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    r, theta, phi = 0.2 + 0.8 * E1, 2 * np.pi * E2, (2 * np.pi if full else np.pi) * E3
    X = (R0 + r * np.cos(theta)) * np.cos(phi)
    Y = (R0 + r * np.cos(theta)) * np.sin(phi)
    Z = r * np.sin(theta)
    return {"eta1": eta1, "eta2": eta2, "eta3": eta3, **{n: (("eta1", "eta2", "eta3"), c) for n, c in zip("XYZ", (X, Y, Z))}}


def scalar(coords, t=None):
    shape = tuple(len(coords[d]) for d in ("eta1", "eta2", "eta3"))
    Z = coords["Z"][1]
    values = np.broadcast_to(Z, shape).copy()
    if t is None:
        return xr.DataArray(values, dims=("eta1", "eta2", "eta3"), coords=coords, name="p", attrs={"label": "p"})
    stacked = np.stack([values * (1 + ti) for ti in t])
    return xr.DataArray(stacked, dims=("t", "eta1", "eta2", "eta3"), coords={"t": t, **coords}, name="p", attrs={"label": "p"})


def plane_coords(n1=10, n2=12):
    """A 2-D run: a square in the x-y plane, eta3 with a single point."""
    eta1, eta2, eta3 = np.linspace(0, 1, n1), np.linspace(0, 1, n2), np.array([0.5])
    E1, E2, _ = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    return {"eta1": eta1, "eta2": eta2, "eta3": eta3, "X": (("eta1", "eta2", "eta3"), 2 * E1 - 1),
            "Y": (("eta1", "eta2", "eta3"), 2 * E2 - 1), "Z": (("eta1", "eta2", "eta3"), 0 * E1)}


def vortex(coords):
    X, Y = coords["X"][1], coords["Y"][1]
    u = np.stack([-Y, X, 0 * X])
    return xr.DataArray(u, dims=("component", "eta1", "eta2", "eta3"), coords={"component": [0, 1, 2], **coords},
                        name="u", attrs={"label": "u"})


def actor_names(plotter):
    return set(plotter.renderer.actors)


def test_structured_grid_carries_scalars_and_vectors_on_physical_points():
    field = scalar(torus_coords())
    grid = p3.structured_grid(field)
    assert grid.dimensions == (6, 12, 16)
    np.testing.assert_allclose(grid.points[:, 2], np.asarray(field.Z).ravel(order="F"))
    np.testing.assert_allclose(grid["p"], np.asarray(field).ravel(order="F"))

    vgrid = p3.structured_grid(vortex(plane_coords()))
    assert vgrid.active_vectors_name == "u"
    assert "|u|" in vgrid.point_data
    with pytest.raises(ValueError, match="select every dimension"):
        p3.structured_grid(scalar(torus_coords(), t=[0.0, 1.0]))
    with pytest.raises(ValueError, match="physical coordinates"):
        p3.structured_grid(field.drop_vars(["X", "Y", "Z"]))


def test_a_selected_away_spatial_dimension_comes_back_flat():
    field = scalar(plane_coords()).isel(eta3=0)
    assert "eta3" not in field.dims
    grid = p3.structured_grid(field)
    assert grid.dimensions == (10, 12, 1)
    assert p3.is_flat(grid)
    with pytest.raises(ValueError, match="at least two"):
        p3.structured_grid(scalar(plane_coords()).isel(eta2=0, eta3=0))


def test_push_forward_recovers_cartesian_components():
    coords = torus_coords(n1=20, n2=40, n3=48)
    X, Y, Z = (coords[n][1] for n in "XYZ")
    E1, E2, E3 = np.meshgrid(coords["eta1"], coords["eta2"], coords["eta3"], indexing="ij")
    r, theta, phi = 0.2 + 0.8 * E1, 2 * np.pi * E2, 2 * np.pi * E3
    R = 3.0 + r * np.cos(theta)
    # the analytic Jacobian dX_a/de_i of torus_coords' mapping
    jac = np.array([
        [0.8 * np.cos(theta) * np.cos(phi), -2 * np.pi * r * np.sin(theta) * np.cos(phi), -2 * np.pi * R * np.sin(phi)],
        [0.8 * np.cos(theta) * np.sin(phi), -2 * np.pi * r * np.sin(theta) * np.sin(phi), 2 * np.pi * R * np.cos(phi)],
        [0.8 * np.sin(theta), 2 * np.pi * r * np.cos(theta), 0 * E1],
    ])
    cartesian = np.stack([-Y, X, 0 * X])  # a toroidal field
    inverse = np.linalg.inv(np.moveaxis(jac, (0, 1), (-2, -1)))
    contravariant = np.einsum("...ia,a...->i...", inverse, cartesian)
    field = vortex(coords).copy(data=contravariant)
    pushed = p3.push_forward(field).values
    # second-order finite differences, including at the edges
    assert np.abs(pushed - cartesian).max() / np.abs(cartesian).max() < 5e-3
    with pytest.raises(ValueError, match="two points"):
        p3.push_forward(vortex(plane_coords()))


def test_boundary_faces_drop_periodic_seams_and_polar_axes():
    def points(coords):
        return np.stack([coords[n][1] for n in "XYZ"], axis=-1)

    # full torus: phi seam dropped, both eta1 faces kept, eta2 (theta) seam dropped
    assert len(p3.boundary_faces(points(torus_coords(full=True)))) == 2
    # half torus: the two phi end caps are real boundaries
    assert len(p3.boundary_faces(points(torus_coords(full=False)))) == 4
    # a 2-D grid is its own face
    assert len(p3.boundary_faces(points(plane_coords()))) == 1


def test_isosurface_is_a_surface_in_3d_and_contour_lines_in_2d():
    plotter = p3.pyvista_isosurface(scalar(torus_coords()), values=3)
    assert "isosurface" in actor_names(plotter) and "plane" not in actor_names(plotter)
    plotter.close()

    flat = scalar(plane_coords()).copy(data=np.asarray(plane_coords()["X"][1]))
    plotter = p3.pyvista_isosurface(flat, values=3)
    assert {"plane", "isosurface"} <= actor_names(plotter)
    assert np.allclose(np.abs(plotter.camera.direction), (0, 0, 1))  # looking at the plane
    plotter.close()


def test_slices_default_to_midplanes_in_3d_and_the_whole_plane_in_2d():
    field = scalar(torus_coords())
    assert [piece.shape for piece in p3.prepare_slices_3d(field)] == [(1, 12, 16), (6, 1, 16), (6, 12, 1)]
    cuts = p3.prepare_slices_3d(field, cuts={"eta3": [0.0, -1], "eta1": 2})
    assert [piece.shape for piece in cuts] == [(6, 12, 1), (6, 12, 1), (1, 12, 16)]
    assert cuts[1].eta3.item() == 1.0
    assert p3.prepare_slices_3d(field, cuts={"eta3": "last"})[0].eta3.item() == 1.0   # still accepted
    with pytest.raises(ValueError, match="eta1, eta2 or eta3"):
        p3.prepare_slices_3d(field, cuts={"t": 0})

    flat = scalar(plane_coords())
    assert [piece.shape for piece in p3.prepare_slices_3d(flat)] == [flat.shape]
    plotter = field.struphy.plot.slices_3d(cuts={"eta3": [0, 0.5]})
    assert {"slice0", "slice1"} <= actor_names(plotter)
    plotter.close()
    assert len(field.struphy.data.slices_3d(cuts={"eta1": 0.5})) == 1


def test_glyphs_and_streamlines_for_a_2d_vector_field():
    flow = vortex(plane_coords())
    plotter = flow.struphy.plot.glyphs(stride=2)
    assert "glyphs" in actor_names(plotter)
    plotter.close()
    plotter = flow.struphy.plot.streamlines(n_points=10)
    assert "streamlines" in actor_names(plotter)
    lines = plotter.renderer.actors["streamlines"].mapper.dataset
    np.testing.assert_allclose(lines.points[:, 2], 0.0, atol=1e-12)  # stays on the plane
    plotter.close()
    with pytest.raises(ValueError, match="cartesian"):
        flow.struphy.plot.glyphs(components="covariant")


def test_orbit_polylines_drop_lost_samples_and_color_by_class():
    t = np.linspace(0, 1, 5)
    x = np.array([[3.0, 3.1], [3.1, 3.2], [3.2, 0.0], [3.3, 0.0], [3.4, 0.0]])
    orbits = xr.Dataset(
        {"x": (("t", "marker"), x), "y": (("t", "marker"), np.where(x == 0, 0, 0.5)),
         "z": (("t", "marker"), np.where(x == 0, 0, 0.1)), "v_par": (("t", "marker"), np.where(x == 0, 0, 1.0))},
        coords={"t": t, "marker": [0, 1]},
    )
    lines = p3.orbit_polylines(orbits)
    assert lines.n_points == 5 + 2  # marker 1 is lost after two samples
    assert lines.n_lines == 2
    plotter = orbits.struphy.plot.orbits_3d(color_by="classification")
    assert {"orbits_passing", "orbits_lost"} <= actor_names(plotter)
    plotter.close()
    with pytest.raises(ValueError, match="color_by"):
        p3.orbit_polylines(orbits, color_by="missing")


def test_domain_wireframe_draws_only_boundary_lines():
    def cylinder(eta1, eta2, eta3, squeeze_out=False):
        E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
        r, theta = E1, 2 * np.pi * E2
        return r * np.cos(theta), r * np.sin(theta), 4 * E3

    plotter = p3.pyvista_domain(cylinder, n1=4, n2=8, n3=3, resolution=2)
    wires = plotter.renderer.actors["wireframe"].mapper.dataset
    assert wires.n_lines > 0
    plotter.close()
    flat = p3.pyvista_domain(cylinder, n1=4, n2=8, n3=1, resolution=2)  # a 2-D cross-section
    assert "wireframe" in actor_names(flat)
    flat.close()


def test_save_movie_writes_one_frame_per_step(tmp_path):
    pytest.importorskip("imageio")
    from PIL import Image

    field = scalar(plane_coords(), t=[0.0, 0.5, 1.0, 1.5])
    path = field.struphy.plot.movie(tmp_path / "movie.gif", kind="isosurface", step=2, values=3)
    assert Image.open(path).n_frames == 2
    with pytest.raises(ValueError, match="kind"):
        p3.save_movie(field, tmp_path / "x.gif", kind="volume")
