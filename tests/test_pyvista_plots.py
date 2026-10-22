"""Tests for the PyVista 3-D views, for 3-D fields and for 2-D ones (one flat direction)."""

import warnings

import numpy as np
import pytest
import xarray as xr

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

import plasma_plots  # noqa: E402, F401  (registers the accessors)
from plasma_plots import pyvista_plots as p3  # noqa: E402


def torus_coords(n1=6, n2=12, n3=16, R0=3.0, full=True):
    eta1, eta2, eta3 = (
        np.linspace(0, 1, n1),
        np.linspace(0, 1, n2),
        np.linspace(0, 1, n3),
    )
    E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    r, theta, phi = 0.2 + 0.8 * E1, 2 * np.pi * E2, (2 * np.pi if full else np.pi) * E3
    X = (R0 + r * np.cos(theta)) * np.cos(phi)
    Y = (R0 + r * np.cos(theta)) * np.sin(phi)
    Z = r * np.sin(theta)
    return {
        "eta1": eta1,
        "eta2": eta2,
        "eta3": eta3,
        **{n: (("eta1", "eta2", "eta3"), c) for n, c in zip("XYZ", (X, Y, Z))},
    }


def scalar(coords, t=None):
    shape = tuple(len(coords[d]) for d in ("eta1", "eta2", "eta3"))
    Z = coords["Z"][1]
    values = np.broadcast_to(Z, shape).copy()
    if t is None:
        return xr.DataArray(
            values,
            dims=("eta1", "eta2", "eta3"),
            coords=coords,
            name="p",
            attrs={"label": "p"},
        )
    stacked = np.stack([values * (1 + ti) for ti in t])
    return xr.DataArray(
        stacked,
        dims=("t", "eta1", "eta2", "eta3"),
        coords={"t": t, **coords},
        name="p",
        attrs={"label": "p"},
    )


def plane_coords(n1=10, n2=12):
    """A 2-D run: a square in the x-y plane, eta3 with a single point."""
    eta1, eta2, eta3 = np.linspace(0, 1, n1), np.linspace(0, 1, n2), np.array([0.5])
    E1, E2, _ = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    return {
        "eta1": eta1,
        "eta2": eta2,
        "eta3": eta3,
        "X": (("eta1", "eta2", "eta3"), 2 * E1 - 1),
        "Y": (("eta1", "eta2", "eta3"), 2 * E2 - 1),
        "Z": (("eta1", "eta2", "eta3"), 0 * E1),
    }


def vortex(coords):
    X, Y = coords["X"][1], coords["Y"][1]
    u = np.stack([-Y, X, 0 * X])
    return xr.DataArray(
        u,
        dims=("component", "eta1", "eta2", "eta3"),
        coords={"component": [0, 1, 2], **coords},
        name="u",
        attrs={"label": "u"},
    )


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
    assert [piece.shape for piece in p3.prepare_slices_3d(field)] == [
        (1, 12, 16),
        (6, 1, 16),
        (6, 12, 1),
    ]
    cuts = p3.prepare_slices_3d(field, cuts={"eta3": [0.0, -1], "eta1": 2})
    assert [piece.shape for piece in cuts] == [(6, 12, 1), (6, 12, 1), (1, 12, 16)]
    assert cuts[1].eta3.item() == 1.0
    assert p3.prepare_slices_3d(field, cuts={"eta3": "last"})[0].eta3.item() == 1.0  # still accepted
    with pytest.raises(ValueError, match="eta1, eta2 or eta3"):
        p3.prepare_slices_3d(field, cuts={"t": 0})

    flat = scalar(plane_coords())
    assert [piece.shape for piece in p3.prepare_slices_3d(flat)] == [flat.shape]
    plotter = field.plasma.plot.slices_3d(cuts={"eta3": [0, 0.5]})
    assert {"slice0", "slice1"} <= actor_names(plotter)
    plotter.close()
    assert len(field.plasma.data.slices_3d(cuts={"eta1": 0.5})) == 1


def test_glyphs_and_streamlines_for_a_2d_vector_field():
    flow = vortex(plane_coords())
    plotter = flow.plasma.plot.glyphs(stride=2)
    assert "glyphs" in actor_names(plotter)
    plotter.close()
    plotter = flow.plasma.plot.streamlines(n_points=10)
    assert "streamlines" in actor_names(plotter)
    lines = plotter.renderer.actors["streamlines"].mapper.dataset
    np.testing.assert_allclose(lines.points[:, 2], 0.0, atol=1e-12)  # stays on the plane
    plotter.close()
    with pytest.raises(ValueError, match="cartesian"):
        flow.plasma.plot.glyphs(components="covariant")


def test_orbit_polylines_drop_lost_samples_and_color_by_class():
    t = np.linspace(0, 1, 5)
    x = np.array([[3.0, 3.1], [3.1, 3.2], [3.2, 0.0], [3.3, 0.0], [3.4, 0.0]])
    orbits = xr.Dataset(
        {
            "x": (("t", "marker"), x),
            "y": (("t", "marker"), np.where(x == 0, 0, 0.5)),
            "z": (("t", "marker"), np.where(x == 0, 0, 0.1)),
            "v_par": (("t", "marker"), np.where(x == 0, 0, 1.0)),
        },
        coords={"t": t, "marker": [0, 1]},
    )
    lines = p3.orbit_polylines(orbits)
    assert lines.n_points == 5 + 2  # marker 1 is lost after two samples
    assert lines.n_lines == 2
    plotter = orbits.plasma.plot.orbits_3d(color_by="classification")
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
    path = field.plasma.plot.movie(tmp_path / "movie.gif", kind="isosurface", step=2, values=3)
    assert Image.open(path).n_frames == 2
    with pytest.raises(ValueError, match="kind"):
        p3.save_movie(field, tmp_path / "x.gif", kind="volume")


def mesh_actors(plotter):
    """The drawn PolyData of each surface actor (not the axes or text)."""
    return [
        actor
        for actor in plotter.renderer.actors.values()
        if isinstance(getattr(actor, "mapper", None), pv.DataSetMapper)
    ]


def square():
    vertices = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    return vertices, np.array([[0, 1, 2], [1, 3, 2]])


def test_surfaces_draw_a_polydata_with_the_first_cycle_color():
    from matplotlib.colors import to_rgb

    plotter = p3.pyvista_surfaces(pv.Sphere(), title="A sphere")
    (actor,) = mesh_actors(plotter)
    assert actor.mapper.dataset.n_cells == pv.Sphere().n_cells
    assert np.allclose(actor.prop.color.float_rgb, to_rgb("C0"))
    assert plotter.legend is None
    plotter.close()
    grid = p3.pyvista_surfaces(pv.ImageData(dimensions=(3, 3, 3)))  # any dataset: its surface
    assert mesh_actors(grid)[0].mapper.dataset.n_cells == 24
    grid.close()


def test_surfaces_draw_a_vertices_faces_pair_face_on():
    vertices, faces = square()
    plotter = p3.pyvista_surfaces((vertices, faces), show_edges=True)
    (actor,) = mesh_actors(plotter)
    assert actor.mapper.dataset.n_cells == 2 and actor.prop.show_edges
    # a flat scene in a z = const plane is viewed along z
    position, focus, _ = plotter.camera_position
    assert np.allclose(position[:2], focus[:2])
    plotter.close()
    with pytest.raises(ValueError, match="faces index"):
        p3.pyvista_surfaces((vertices, faces + 2))
    with pytest.raises(TypeError, match="a mesh is"):
        p3.pyvista_surfaces("sphere")


def test_surfaces_take_an_object_with_vertices_and_cells_and_skip_empty_ones():
    class SurfaceMesh:  # like simgeometry.mesh.SurfaceMesh
        def __init__(self, vertices, cells):
            self.vertices, self.cells = vertices, cells

    empty = SurfaceMesh(np.empty((0, 3)), np.empty((0, 3), dtype=int))
    plotter = p3.pyvista_surfaces([SurfaceMesh(*square()), empty])
    assert len(mesh_actors(plotter)) == 1
    plotter.close()


def test_a_dict_of_surfaces_gets_labels_colors_and_a_legend():
    meshes = {"inner": pv.Sphere(radius=0.5), "outer": pv.Sphere()}
    plotter = p3.pyvista_surfaces(
        meshes, colors={"inner": "red", "outer": "blue"}, opacity={"inner": 1.0, "outer": 0.3}
    )
    actors = mesh_actors(plotter)
    assert [a.prop.color.float_rgb for a in actors] == [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    assert [a.prop.opacity for a in actors] == [1.0, 0.3]
    assert plotter.legend is not None
    plotter.close()
    no_legend = p3.pyvista_surfaces(meshes, legend=False)
    assert no_legend.legend is None
    no_legend.close()
    with pytest.raises(ValueError, match="one entry per mesh"):
        p3.pyvista_surfaces([pv.Sphere(), pv.Cube()], labels=["only one"])
    with pytest.raises(ValueError, match="no entry for"):
        p3.pyvista_surfaces(meshes, colors={"inner": "red"})


def test_surfaces_combine_with_an_existing_plotter():
    plotter = p3.pyvista_surfaces({"a": pv.Sphere(), "b": pv.Cube()})
    camera = plotter.camera_position
    same = p3.pyvista_surfaces(pv.Cone(), plotter=plotter)
    assert same is plotter and len(mesh_actors(plotter)) == 3
    assert plotter.camera_position == camera
    plotter.close()


def test_surfaces_colored_by_scalars_share_one_scale():
    sphere, cube = pv.Sphere(), pv.Cube()
    plotter = p3.pyvista_surfaces([sphere, cube], scalars=lambda points: points[:, 2])
    assert list(plotter.scalar_bars.keys()) == ["value"]
    ranges = {tuple(actor.mapper.scalar_range) for actor in mesh_actors(plotter)}
    assert ranges == {(-0.5, 0.5)}  # the cube's z range covers the sphere's
    assert "value" not in sphere.point_data  # the caller's meshes are left as they are
    plotter.close()

    def potential(points):
        return np.linalg.norm(points, axis=1)

    named = p3.pyvista_surfaces(sphere, scalars=potential, clim=(0.0, 1.0), cmap="plasma")
    assert list(named.scalar_bars.keys()) == ["potential"]
    named.close()
    sphere.point_data["phi"] = sphere.points[:, 0]
    by_name = p3.pyvista_surfaces(sphere, scalars="phi")
    assert list(by_name.scalar_bars.keys()) == ["phi"]
    by_name.close()
    with pytest.raises(ValueError, match="no point data 'phi'"):
        p3.pyvista_surfaces({"s": sphere, "t": pv.Cube()}, scalars="phi")


def test_surfaces_without_values_take_no_part_in_the_scale():
    valued, empty = pv.Sphere(), pv.Sphere(center=(2, 0, 0))
    valued.point_data["voltage"] = np.linspace(0.0, 5.0, valued.n_points)
    empty.point_data["voltage"] = np.full(empty.n_points, np.nan)  # e.g. not a conductor
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        plotter = p3.pyvista_surfaces({"conductor": valued, "probe": empty}, scalars="voltage")
    ranges = {tuple(actor.mapper.scalar_range) for actor in mesh_actors(plotter)}
    assert ranges == {(0.0, 5.0)}
    plotter.close()


def point_actors(plotter):
    """The actors that draw vertices only (point sets), not surfaces or arrows."""
    return [actor for actor in mesh_actors(plotter) if actor.mapper.dataset.n_points == actor.mapper.dataset.n_verts]


def test_points_draw_a_set_with_the_first_cycle_color():
    from matplotlib.colors import to_rgb

    points = np.random.default_rng(0).random((20, 3))
    plotter = p3.pyvista_points(points, size=5, title="Samples")
    (actor,) = mesh_actors(plotter)
    assert np.allclose(actor.mapper.dataset.points, points)
    assert np.allclose(actor.prop.color.float_rgb, to_rgb("C0")) and actor.prop.point_size == 5
    assert plotter.legend is None
    plotter.close()
    with pytest.raises(ValueError, match="must have shape"):
        p3.pyvista_points(np.zeros((4, 5)))


def test_a_dict_of_points_gets_labels_colors_and_a_legend():
    from matplotlib.colors import to_rgb

    plotter = p3.pyvista_points(
        {"inside": np.zeros((4, 3)), "outside": np.ones((3, 3)), "none": np.empty((0, 3))},
        opacity={"inside": 1.0, "outside": 0.5, "none": 1.0},
    )
    actors = mesh_actors(plotter)
    assert [a.mapper.dataset.n_points for a in actors] == [4, 3]  # the empty set is skipped
    assert [a.prop.color.float_rgb for a in actors] == [to_rgb("C0"), to_rgb("C1")]
    assert [a.prop.opacity for a in actors] == [1.0, 0.5]
    assert plotter.legend.GetNumberOfEntries() == 2
    plotter.close()
    with pytest.raises(ValueError, match="one entry per point set"):
        p3.pyvista_points([np.zeros((2, 3)), np.ones((2, 3))], labels=["only one"])


def test_points_take_component_major_and_planar_arrays():
    xyz = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0], [5.0, 5.0, 5.0, 5.0]])  # (3, n)
    plotter = p3.pyvista_points(xyz)
    assert np.allclose(mesh_actors(plotter)[0].mapper.dataset.points, xyz.T)
    # a flat scene in a z = const plane is viewed along z
    position, focus, _ = plotter.camera_position
    assert np.allclose(position[:2], focus[:2])
    plotter.close()
    planar = p3.pyvista_points([xyz[:2], xyz[:2].T + 1])  # (2, n) and (n, 2): in z = 0
    first, second = (actor.mapper.dataset.points for actor in mesh_actors(planar))
    assert np.allclose(first, np.column_stack([xyz[:2].T, np.zeros(4)]))
    assert np.allclose(second[:, 2], 0.0)
    planar.close()


def test_points_with_vectors_draw_arrows_of_the_set_color():
    from matplotlib.colors import to_rgb

    sphere = pv.Sphere(theta_resolution=8, phi_resolution=8)
    normals = np.asarray(sphere.point_normals)
    plotter = p3.pyvista_points({"nodes": sphere.points}, vectors={"nodes": normals.T}, vector_scale=0.2)
    (points,) = point_actors(plotter)
    (arrows,) = [actor for actor in mesh_actors(plotter) if actor is not points]
    assert np.allclose(arrows.prop.color.float_rgb, to_rgb("C0"))
    # unit normals: every arrow is vector_scale long, so they reach 0.2 beyond the unit sphere
    assert np.isclose(np.linalg.norm(arrows.mapper.dataset.points, axis=1).max(), 0.5 + 0.2, atol=1e-3)
    plotter.close()
    default = p3.pyvista_points(sphere.points, vectors=normals)  # 5% of the scene's extent
    assert len(mesh_actors(default)) == 2
    default.close()
    with pytest.raises(ValueError, match="vectors of 'nodes'"):
        p3.pyvista_points({"nodes": sphere.points}, vectors={"nodes": normals[:3]})
    with pytest.raises(ValueError, match="for no point set"):
        p3.pyvista_points({"nodes": sphere.points}, vectors={"other": normals})


def test_points_and_surfaces_in_one_plotter_share_the_legend_and_color_cycle():
    from matplotlib.colors import to_rgb

    plotter = p3.pyvista_surfaces({"vessel": pv.Sphere()}, opacity=0.3)
    assert plotter.legend is None  # one entry: no legend yet
    camera = plotter.camera_position
    same = p3.pyvista_points({"samples": np.zeros((3, 3)), "nodes": np.ones((2, 3))}, plotter=plotter)
    assert same is plotter and plotter.camera_position == camera
    assert plotter.legend.GetNumberOfEntries() == 3
    colors = [actor.prop.color.float_rgb for actor in point_actors(plotter)]
    assert np.allclose(colors, [to_rgb("C1"), to_rgb("C2")])  # the cycle continues after the vessel
    p3.pyvista_surfaces({"coil": pv.Cube()}, plotter=plotter)
    assert plotter.legend.GetNumberOfEntries() == 4
    p3.pyvista_points({"hidden": np.zeros((1, 3))}, legend=False, plotter=plotter)
    assert plotter.legend.GetNumberOfEntries() == 4
    plotter.close()


def test_points_colored_by_scalars_share_one_scale():
    a, b = np.zeros((3, 3)), np.ones((4, 3))
    a[:, 2] = [0.0, 1.0, 2.0]
    plotter = p3.pyvista_points([a, b], scalars=lambda points: points[:, 2], vectors=[a + 1, None])
    assert list(plotter.scalar_bars.keys()) == ["value"]
    ranges = {tuple(actor.mapper.scalar_range) for actor in mesh_actors(plotter)}
    assert ranges == {(0.0, 2.0)}  # both sets and the arrows
    assert plotter.legend is None
    plotter.close()

    def height(points):
        return points[:, 2]

    named = p3.pyvista_points({"a": a, "b": b}, scalars={"a": [0, 1, np.nan], "b": [1, 2, 3, 4]}, cmap="plasma")
    assert {tuple(actor.mapper.scalar_range) for actor in mesh_actors(named)} == {(0.0, 4.0)}
    assert named.legend is None
    named.close()
    by_function = p3.pyvista_points(a, scalars=height, clim=(-1.0, 1.0))
    assert list(by_function.scalar_bars.keys()) == ["height"]
    assert tuple(mesh_actors(by_function)[0].mapper.scalar_range) == (-1.0, 1.0)
    by_function.close()
    with pytest.raises(ValueError, match="one value per point"):
        p3.pyvista_points(a, scalars=[1.0, 2.0])
