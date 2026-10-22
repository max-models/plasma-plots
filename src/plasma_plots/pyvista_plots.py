"""Three-dimensional PyVista views of labeled Struphy output.

Every function returns a ``pyvista.Plotter`` without showing it: call ``.show()`` in an
interactive session or notebook, or ``.screenshot(path)`` (with ``pyvista.OFF_SCREEN = True``
in batch jobs). Pass ``plotter=`` to draw several views into one scene.

:func:`pyvista_surfaces` draws any surface meshes (geometry, marching-cubes surfaces) in the
same style. :func:`scene` shows a plotter as an interactive 3-D scene in a notebook, which docs sites
built from notebooks embed as they are.

Fields need their physical ``X``, ``Y``, ``Z`` coordinates attached, as every Struphy field
product has; orbits need physical positions ``x``, ``y``, ``z``.
"""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from .analysis import ORBIT_CLASSES, classify_orbits
from .arrays import close_periodic, logical_dims, mapping_jacobian, validate_array, value_label
from .mpi import rank_zero

ORBIT_CLASS_COLORS = {"passing": "tab:blue", "trapped": "tab:orange", "lost": "grey"}


def _pv():
    try:
        import pyvista
    except ImportError as error:  # pragma: no cover
        raise ImportError('PyVista plots need the optional extra: pip install "plasma-plots[pyvista]"') from error
    return pyvista


def _label(data):
    return data.attrs.get("label") or data.attrs.get("long_name") or data.name or "value"


def _plotter(plotter):
    return _pv().Plotter() if plotter is None else plotter


def _spatial(data: xr.DataArray, *, extra=()) -> xr.DataArray:
    """``data`` transposed to ``(*extra, eta1, eta2, eta3)``, checking nothing else is left.

    A spatial dimension that was selected away (e.g. ``field.isel(eta3=0)`` of a 2-D run) comes
    back with a single point, so a plane is still a (flat) structured grid.
    """
    validate_array(data, required_dims=extra)
    spatial = logical_dims(data)
    others = set(data.dims) - {*extra, *spatial}
    if others:
        raise ValueError(f"select every dimension except {(*extra, *spatial)} first; {sorted(others)} remain")
    missing = [name for name in ("X", "Y", "Z") if name not in data.coords]
    if missing:
        raise ValueError(f"3-D views need physical coordinates X, Y, Z on {data.name!r}; missing {missing}")
    absent = [dim for dim in spatial if dim not in data.dims]
    if len(absent) > 1:
        raise ValueError(f"3-D views need at least two of {spatial}; {data.name!r} has dims {data.dims}")
    for dim in absent:
        data = data.expand_dims(dim) if dim in data.coords else data.expand_dims({dim: [0.0]})
    coords = {}
    for name in ("X", "Y", "Z"):
        coordinate = data.coords[name]
        for dim in spatial:
            if dim not in coordinate.dims:
                coordinate = coordinate.expand_dims({dim: data.sizes[dim]})
        coords[name] = coordinate.transpose(*spatial).variable
    return data.assign_coords(coords).transpose(*extra, *spatial)


def is_flat(grid) -> bool:
    """Whether a structured grid has a single point in one logical direction (a 2-D run or cut).

    Parameters
    ----------
    grid : pyvista.StructuredGrid
        The grid, e.g. from :func:`structured_grid`.

    Returns
    -------
    bool
        ``True`` if any of the grid's dimensions is 1.
    """
    return 1 in tuple(grid.dimensions)


def _plane_normal(grid):
    """The normal of a flat grid that lies in one physical plane, else ``None``."""
    points = np.asarray(grid.points, dtype=float)
    centered = points - points.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular[-1] > 1e-6 * max(singular[0], 1e-300):
        return None
    return vt[-1]


def _camera(plotter, grid):
    """Look straight at a planar grid (a 2-D run) or surface; leave 3-D scenes at the default view."""
    if isinstance(grid, _pv().StructuredGrid) and not is_flat(grid):
        return
    if not grid.n_points:
        return
    normal = _plane_normal(grid)
    if normal is None:
        return
    axis = int(np.abs(normal).argmax())
    if abs(normal[axis]) > 0.999:  # an axis-aligned plane: keep both in-plane axes pointing right/up
        (plotter.view_yz, plotter.view_xz, plotter.view_xy)[axis]()
    else:
        up = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (0.0, 1.0, 0.0)
        plotter.view_vector(tuple(normal if normal[axis] > 0 else -normal), viewup=up)
    plotter.reset_camera()


def _points(data: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.asarray(data.coords[name].transpose(*logical_dims(data)), dtype=float) for name in ("X", "Y", "Z"))


def structured_grid(data: xr.DataArray, *, name: str | None = None):
    """A ``pyvista.StructuredGrid`` of a selected ``(eta1, eta2, eta3)`` field on its physical points.

    A scalar field becomes point data ``name`` (default: the field's label); a vector field with
    a ``component`` dimension of three Cartesian components becomes point vectors ``name``, plus
    their magnitude as ``"|name|"``. Periodic directions are closed, see :func:`plasma_plots.arrays.close_periodic`.
    The grid is useful directly for any PyVista filter.

    Parameters
    ----------
    data : xarray.DataArray
        A field with dims ``(eta1, eta2, eta3)`` (one of them may be selected away), or
        ``(component, eta1, eta2, eta3)`` for a vector field, and physical coordinates ``X``,
        ``Y``, ``Z``.
    name : str, optional
        The name of the point data. Default: the field's label.

    Returns
    -------
    pyvista.StructuredGrid
        The grid with the field as its active scalars or vectors.

    Raises
    ------
    ValueError
        If other dimensions remain, the physical coordinates are missing, fewer than two
        logical dimensions are left, or a vector field doesn't have three components.

    Examples
    --------
    >>> grid = structured_grid(phi.isel(t=-1))
    >>> grid.contour([0.0]).plot()
    """
    pv = _pv()
    vector = "component" in data.dims
    data = close_periodic(_spatial(data, extra=("component",) if vector else ()), logical_dims(data))
    grid = pv.StructuredGrid(*_points(data))
    name = name or _label(data)
    if vector:
        if data.sizes["component"] != 3:
            raise ValueError(f"a vector field needs 3 components; got {data.sizes['component']}")
        vectors = np.stack([np.asarray(c).ravel(order="F") for c in data], axis=-1)
        grid.point_data[name] = vectors
        grid.point_data[f"|{name}|"] = np.linalg.norm(vectors, axis=1)
        grid.set_active_vectors(name)
    else:
        grid.point_data[name] = np.asarray(data).ravel(order="F")
    return grid


def push_forward(data: xr.DataArray) -> xr.DataArray:
    """Cartesian components of a vector field given by contravariant logical components.

    ``data`` has dims ``(component, eta1, eta2, eta3)``, e.g. from
    ``out.evaluate(name, eta1=..., eta2=..., eta3=..., representation="v")``. The Cartesian
    field is ``sum_i v^i dX/de_i``, with the Jacobian of the mapping differentiated numerically
    from the attached ``X``, ``Y``, ``Z`` coordinates, so every logical direction needs at least
    two points.

    Parameters
    ----------
    data : xarray.DataArray
        The vector field, dims ``(component, eta1, eta2, eta3)`` with three contravariant
        components and physical coordinates ``X``, ``Y``, ``Z``.

    Returns
    -------
    xarray.DataArray
        The Cartesian ``x``, ``y``, ``z`` components, same dims and coordinates, labeled
        ``"<label> (Cartesian)"``.

    Raises
    ------
    ValueError
        If the field doesn't have three components or a logical direction has fewer than two
        points.

    Examples
    --------
    >>> B_xyz = push_forward(
    ...     out.evaluate("em_fields/b_field", representation="v").isel(t=-1)
    ... )
    """
    data = _spatial(data, extra=("component",))
    if data.sizes["component"] != 3:
        raise ValueError(f"a vector field needs 3 components; got {data.sizes['component']}")
    short = [dim for dim in logical_dims(data) if data.sizes[dim] < 2]
    if short:
        raise ValueError(f"pushing forward needs at least two points along {short}")
    jacobian = mapping_jacobian(data)  # jacobian[a, i] = dX_a / de_i
    cartesian = np.einsum("ai...,i...->a...", jacobian, np.asarray(data))
    out = data.copy(data=cartesian)
    out.attrs = {**data.attrs, "label": f"{_label(data)} (Cartesian)"}
    return out


def _vector_grid(data, components, name):
    if components == "contravariant":
        data = push_forward(data)
    elif components != "cartesian":
        raise ValueError(f'components must be "cartesian" or "contravariant"; got {components!r}')
    return structured_grid(data, name=name)


def _vtk_text(text: str) -> str:
    """Mathtext that VTK renders: inside ``$...$`` VTK reads ``|`` as a table-cell separator, so
    it becomes ``\\vert`` (``$|B|$`` shows as |B|, not as the raw string)."""
    import re

    return re.sub(r"\$[^$]*\$", lambda math: math.group(0).replace("|", r"\vert "), text)


def _bar(title):
    return {"title": _vtk_text(title), "fmt": "%.3g"}


def _clim(values, clim, *, symmetric=False, robust=False):
    if clim is not None:
        return tuple(clim)
    from .plotting import color_limits

    return color_limits(values, symmetric=symmetric, robust=robust)


def _face(points, axis, index):
    face = [slice(None)] * 3
    face[axis] = slice(index, index + 1) if index >= 0 else slice(index, None)
    return points[tuple(face)]


def boundary_keys(points: np.ndarray) -> list[tuple[int, int]]:
    """Find the logical faces of a ``(n1, n2, n3, 3)`` point array that are real boundaries.

    Faces that collapse to a line or point (a polar axis) and pairs of opposite faces that
    coincide (the seam of a periodic direction, e.g. ``phi = 0`` of a full torus) are dropped.
    A grid that is flat in one direction (a 2-D run) is its own single face.

    Parameters
    ----------
    points : numpy.ndarray
        Physical points, shape ``(n1, n2, n3, 3)``.

    Returns
    -------
    list of (int, int)
        ``(axis, index)`` of each boundary face; ``index`` is ``0`` or ``-1``.
    """
    flat = [axis for axis in range(3) if points.shape[axis] == 1]
    if flat:
        return [(flat[0], 0)]
    scale = max(float(np.ptp(points.reshape(-1, 3), axis=0).max()), 1e-300)
    keys = []
    for axis in range(3):
        low, high = _face(points, axis, 0), _face(points, axis, -1)
        if np.allclose(low, high, atol=1e-9 * scale):
            continue
        for index, face in ((0, low), (-1, high)):
            spread = np.sort(np.ptp(face.reshape(-1, 3), axis=0))
            if spread[1] > 1e-9 * scale:  # spans a surface, not a line or point
                keys.append((axis, index))
    return keys


def boundary_faces(points: np.ndarray) -> list[np.ndarray]:
    """Return the real boundary faces of a point array, see :func:`boundary_keys`.

    Parameters
    ----------
    points : numpy.ndarray
        Physical points, shape ``(n1, n2, n3, 3)``.

    Returns
    -------
    list of numpy.ndarray
        The points of each face, with a size-one axis in the face's logical direction.
    """
    return [_face(points, axis, index) for axis, index in boundary_keys(points)]


def _face_lines(face: np.ndarray, resolution: int):
    """Grid lines of a face (a point array with one size-one axis), every ``resolution`` samples."""
    face = face.squeeze(axis=[a for a in range(3) if face.shape[a] == 1][0])
    lines = []
    for along in (0, 1):
        n_other = face.shape[1 - along]
        picks = sorted({*range(0, n_other, resolution), n_other - 1})
        for k in picks:
            line = face[:, k] if along == 0 else face[k, :]
            if len(line) > 1 and np.ptp(line, axis=0).max() > 0:  # skip collapsed lines
                lines.append(line)
    return lines


def _grid_points(grid) -> np.ndarray:
    return np.asarray(grid.points).reshape((*grid.dimensions, 3), order="F")


def _add_context(plotter, grid, show_domain, drawn=()):
    """The domain's outer faces, translucent; not those a drawn surface (``drawn``, point arrays
    like :func:`boundary_faces`' ) already covers, which would flicker against it."""
    if not show_domain:
        return
    pv = _pv()
    faces = [
        face
        for face in boundary_faces(_grid_points(grid))
        if not any(points.shape == face.shape and np.allclose(points, face) for points in drawn)
    ]
    for i, face in enumerate(faces):
        surface = pv.StructuredGrid(face[..., 0], face[..., 1], face[..., 2])
        plotter.add_mesh(surface, color="lightgrey", opacity=0.12, name=f"domain{i}")


def _finish(plotter, title, grid=None):
    """Title and axes; with ``grid`` (a plotter this module created), aim the camera at it."""
    if title:
        plotter.add_text(_vtk_text(title), font_size=10, name="title")
    plotter.show_axes()
    if grid is not None:
        _camera(plotter, grid)
    return plotter


@rank_zero
def pyvista_isosurface(
    data: xr.DataArray,
    *,
    values: int | list[float] = 5,
    cmap="viridis",
    opacity: float = 1.0,
    clim=None,
    show_domain: bool = True,
    title: str | None = None,
    symmetric: bool = False,
    robust: bool = False,
    plotter=None,
):
    """Contour surfaces of a selected scalar ``(eta1, eta2, eta3)`` field in physical space.

    ``values`` is the number of evenly spaced levels, or explicit levels (between the color
    limits, which ``symmetric``/``robust`` set as in :func:`~plasma_plots.plotting.color_limits`).
    ``show_domain`` draws
    the domain's outer surface translucently for context. For a 2-D field (one logical
    direction with a single point) the levels are contour lines over the colored plane.

    Parameters
    ----------
    data : xarray.DataArray
        A scalar field with dims ``(eta1, eta2, eta3)`` (every other dimension selected; one
        logical dimension may be selected away) and physical coordinates ``X``, ``Y``, ``Z``.
    values : int or list of float, optional
        The number of evenly spaced levels strictly between the color limits, or explicit
        levels. Default: 5.
    cmap : str or matplotlib colormap, optional
        The colormap. Default: ``"viridis"``.
    opacity : float, optional
        Opacity of the surfaces (of the plane for a 2-D field). Default: 1.
    clim : (float, float), optional
        Color limits; default from the field's values, see ``symmetric`` and ``robust``.
    show_domain : bool, optional
        Draw the domain's outer surface translucently for context (3-D fields only).
        Default: ``True``.
    title : str, optional
        Text in the scene's corner. Default: the field's label; ``""`` for none.
    symmetric : bool, optional
        Color limits symmetric about zero. Default: ``False``.
    robust : bool, optional
        Color limits from percentiles instead of the extremes, so outliers don't wash out the
        colors. Default: ``False``.
    plotter : pyvista.Plotter, optional
        Draw into this scene instead of a new one, to combine several views. The camera is only
        aimed at the field when this function creates the plotter.

    Returns
    -------
    pyvista.Plotter
        The scene, not yet shown: call ``.show()`` or ``.screenshot(path)``.

    Raises
    ------
    ValueError
        If other dimensions remain or the physical coordinates are missing.

    See Also
    --------
    pyvista_slices : The field on surfaces of constant logical coordinate.

    Examples
    --------
    >>> pyvista_isosurface(phi.isel(t=-1), values=[-0.1, 0.1]).show()
    >>> pyvista_isosurface(phi.isel(t=-1), symmetric=True, opacity=0.6).show()
    """
    grid = structured_grid(data)
    name = grid.active_scalars_name
    lo, hi = _clim(grid[name], clim, symmetric=symmetric, robust=robust)
    levels = np.linspace(lo, hi, values + 2)[1:-1] if isinstance(values, int) else values
    own = plotter is None
    plotter = _plotter(plotter)
    contours = grid.contour(isosurfaces=list(levels), scalars=name)
    bar = _bar(value_label(data))
    if is_flat(grid):
        plotter.add_mesh(
            grid,
            scalars=name,
            cmap=cmap,
            clim=(lo, hi),
            opacity=opacity,
            scalar_bar_args=bar,
            name="plane",
        )
        if contours.n_points:
            plotter.add_mesh(contours, color="black", line_width=1.5, name="isosurface")
    else:
        _add_context(plotter, grid, show_domain)
        if contours.n_points:
            plotter.add_mesh(
                contours,
                scalars=name,
                cmap=cmap,
                clim=(lo, hi),
                opacity=opacity,
                scalar_bar_args=bar,
                name="isosurface",
            )
    return _finish(plotter, _label(data) if title is None else title, grid if own else None)


def _cut_indices(data, cuts):
    indices = {}
    for dim, positions in (cuts or {}).items():
        if dim not in logical_dims(data):
            first, second, third = logical_dims(data)
            raise ValueError(f"cuts are along {first}, {second} or {third}; got {dim!r}")
        coordinate = np.asarray(data[dim], dtype=float)
        chosen = []
        # not via numpy: a list like [0.0, -1] would turn the index -1 into the coordinate -1.0
        for position in positions if isinstance(positions, (list, tuple)) else [positions]:
            if position in (
                "first",
                "last",
            ):  # accepted, but integer indices are the documented form
                chosen.append(0 if position == "first" else len(coordinate) - 1)
            elif isinstance(position, (bool, str)):
                raise TypeError(
                    f"cannot cut {dim} at {position!r}; use an integer index (e.g. -1) or a float coordinate"
                )
            elif isinstance(position, (int, np.integer)):
                chosen.append(position % len(coordinate))
            else:
                chosen.append(int(np.abs(coordinate - float(position)).argmin()))
        indices[dim] = chosen
    return indices


def prepare_slices_3d(data: xr.DataArray, *, cuts: dict | None = None) -> list[xr.DataArray]:
    """The logical cuts of a scalar ``(eta1, eta2, eta3)`` field that :func:`pyvista_slices` draws.

    ``cuts`` maps ``eta1``/``eta2``/``eta3`` to one position or a list: a float is the nearest
    logical coordinate, an integer a grid index (``-1`` the last). The default is
    the middle of every dimension with more than one point, or for a 2-D field (one dimension
    with a single point) the whole plane. Each cut keeps its size-one dimension, so it still
    maps onto a surface in physical space.

    Parameters
    ----------
    data : xarray.DataArray
        A scalar field with dims ``(eta1, eta2, eta3)`` and physical coordinates ``X``, ``Y``,
        ``Z``.
    cuts : dict, optional
        ``{dim: position or list of positions}`` along ``eta1``, ``eta2``, ``eta3``.

    Returns
    -------
    list of xarray.DataArray
        One array per cut, in the order of ``cuts``.

    Raises
    ------
    ValueError
        If a cut is along another dimension.
    TypeError
        If a position is neither an integer nor a float.
    """
    data = _spatial(data)
    if cuts is None:
        if 1 in data.shape:
            return [data]
        cuts = {dim: data.sizes[dim] // 2 for dim in logical_dims(data)}
    return [data.isel({dim: [index]}) for dim, indices in _cut_indices(data, cuts).items() for index in indices]


@rank_zero
def pyvista_slices(
    data: xr.DataArray,
    *,
    cuts: dict | None = None,
    cmap="viridis",
    clim=None,
    show_domain: bool = True,
    title: str | None = None,
    symmetric: bool = False,
    robust: bool = False,
    plotter=None,
):
    """Surfaces of constant logical coordinate through a scalar field, drawn in physical space.

    On a mapped domain these are the natural cuts: ``cuts={"eta3": [0, 0.25]}`` gives poloidal
    cross-sections of a torus, ``cuts={"eta1": 0.8}`` the field on one flux surface. See
    :func:`prepare_slices_3d` for ``cuts``; color limits are shared by every cut.

    Parameters
    ----------
    data : xarray.DataArray
        A scalar field with dims ``(eta1, eta2, eta3)`` (every other dimension selected; one
        logical dimension may be selected away) and physical coordinates ``X``, ``Y``, ``Z``.
    cuts : dict, optional
        ``{dim: position or list of positions}`` along ``eta1``, ``eta2``, ``eta3``: a float is
        the nearest logical coordinate, an integer a grid index (``-1`` the last). Default: the
        middle of every dimension, or the whole plane of a 2-D field.
    cmap : str or matplotlib colormap, optional
        The colormap. Default: ``"viridis"``.
    clim : (float, float), optional
        Color limits, shared by every cut; default from the whole field's values, see
        ``symmetric`` and ``robust``.
    show_domain : bool, optional
        Draw the domain's outer surface translucently for context. Not drawn when the whole plane of a
        2-D field is shown. Default: ``True``.
    title : str, optional
        Text in the scene's corner. Default: the field's label; ``""`` for none.
    symmetric : bool, optional
        Color limits symmetric about zero. Default: ``False``.
    robust : bool, optional
        Color limits from percentiles instead of the extremes, so outliers don't wash out the
        colors. Default: ``False``.
    plotter : pyvista.Plotter, optional
        Draw into this scene instead of a new one, to combine several views. The camera is only
        aimed at the field when this function creates the plotter.

    Returns
    -------
    pyvista.Plotter
        The scene, not yet shown: call ``.show()`` or ``.screenshot(path)``.

    Raises
    ------
    ValueError
        If other dimensions remain, the physical coordinates are missing, or a cut is along
        another dimension.

    See Also
    --------
    prepare_slices_3d : The cuts, without drawing them.
    pyvista_isosurface : Contour surfaces of the field.

    Examples
    --------
    >>> pyvista_slices(phi.isel(t=-1), cuts={"eta3": [0, 0.25]}).show()
    >>> pyvista_slices(phi.isel(t=-1), cuts={"eta1": 0.8}, symmetric=True).show()
    """
    data = _spatial(data)
    pieces = prepare_slices_3d(data, cuts=cuts)
    lo, hi = _clim(np.asarray(data), clim, symmetric=symmetric, robust=robust)
    own = plotter is None
    plotter = _plotter(plotter)
    grid = structured_grid(data)
    surfaces = [structured_grid(piece, name=_label(data)) for piece in pieces]
    if not (is_flat(grid) and len(pieces) == 1 and pieces[0].shape == data.shape):
        _add_context(
            plotter,
            grid,
            show_domain,
            drawn=[_grid_points(surface) for surface in surfaces],
        )
    for i, surface in enumerate(surfaces):
        plotter.add_mesh(
            surface,
            cmap=cmap,
            clim=(lo, hi),
            line_width=3,  # cuts through a 2-D field are lines
            ambient=0.6,  # keep colors readable on cuts seen at grazing angles
            diffuse=0.45,
            specular=0.0,
            scalar_bar_args=_bar(value_label(data)),
            name=f"slice{i}",
        )
    return _finish(plotter, _label(data) if title is None else title, grid if own else None)


@rank_zero
def pyvista_glyphs(
    data: xr.DataArray,
    *,
    components: Literal["cartesian", "contravariant"] = "cartesian",
    stride: int = 2,
    scale: float | None = None,
    cmap="viridis",
    show_domain: bool = True,
    title: str | None = None,
    plotter=None,
):
    """Arrows of a selected ``(component, eta1, eta2, eta3)`` vector field, colored by magnitude.

    ``components="cartesian"`` (default) for x/y/z components, e.g. a ``*_phy`` product;
    ``"contravariant"`` for logical components, pushed forward by :func:`push_forward`.
    ``stride`` thins the grid in every direction; ``scale`` is the arrow length of the largest
    vector (default: a tenth of the domain size).

    Parameters
    ----------
    data : xarray.DataArray
        A vector field with dims ``(component, eta1, eta2, eta3)`` (three components, every
        other dimension selected) and physical coordinates ``X``, ``Y``, ``Z``.
    components : {"cartesian", "contravariant"}, optional
        How to read the components: Cartesian x/y/z, or contravariant logical components,
        pushed forward. Default: ``"cartesian"``.
    stride : int, optional
        Draw an arrow at every ``stride``-th point in every direction. Default: 2.
    scale : float, optional
        Arrow length of the largest vector, in physical units. Default: a tenth of the domain
        size.
    cmap : str or matplotlib colormap, optional
        The colormap of the magnitude. Default: ``"viridis"``.
    show_domain : bool, optional
        Draw the domain's outer surface translucently for context. Default: ``True``.
    title : str, optional
        Text in the scene's corner. Default: the field's label; ``""`` for none.
    plotter : pyvista.Plotter, optional
        Draw into this scene instead of a new one, to combine several views. The camera is only
        aimed at the field when this function creates the plotter.

    Returns
    -------
    pyvista.Plotter
        The scene, not yet shown: call ``.show()`` or ``.screenshot(path)``.

    Raises
    ------
    ValueError
        If ``stride`` is less than 1, ``components`` is unknown, or the field isn't a
        three-component field over the logical dimensions with physical coordinates.

    See Also
    --------
    pyvista_streamlines : Field lines of the vector field.
    push_forward : Cartesian components from contravariant ones.

    Examples
    --------
    >>> pyvista_glyphs(B.isel(t=-1), stride=3).show()
    >>> pyvista_glyphs(B.isel(t=-1), components="contravariant", scale=0.2).show()
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    name = _label(data)
    full = _vector_grid(data, components, name)
    thinned = _vector_grid(
        data.isel({dim: slice(None, None, stride) for dim in logical_dims(data)}),
        components,
        name,
    )
    magnitude = thinned[f"|{name}|"]
    peak = float(magnitude.max()) if magnitude.size else 0.0
    length = 0.1 * full.length if scale is None else scale
    own = plotter is None
    plotter = _plotter(plotter)
    _add_context(plotter, full, show_domain)
    if peak > 0:
        arrows = thinned.glyph(orient=name, scale=f"|{name}|", factor=length / peak)
        plotter.add_mesh(
            arrows,
            scalars=f"|{name}|",
            cmap=cmap,
            scalar_bar_args=_bar(f"|{name}|"),
            name="glyphs",
        )
    return _finish(plotter, name if title is None else title, full if own else None)


@rank_zero
def pyvista_streamlines(
    data: xr.DataArray,
    *,
    components: Literal["cartesian", "contravariant"] = "cartesian",
    n_points: int = 100,
    source_radius: float | None = None,
    source_center=None,
    max_length: float | None = None,
    tube_radius: float | None = None,
    cmap="viridis",
    show_domain: bool = True,
    title: str | None = None,
    plotter=None,
):
    """Field lines of a selected vector field, e.g. magnetic field lines, colored by magnitude.

    Lines are traced in both directions from ``n_points`` seeds: by default grid points drawn
    at random (reproducibly) across the domain, or, given ``source_center`` and/or
    ``source_radius``, points in that sphere (radius default: a quarter of the domain size;
    center default: the bounding-box center, which can lie outside a curved domain). For a 2-D
    field the lines stay on the plane (the out-of-plane component is ignored). See
    :func:`pyvista_glyphs` for ``components``.

    Parameters
    ----------
    data : xarray.DataArray
        A vector field with dims ``(component, eta1, eta2, eta3)`` (three components, every
        other dimension selected) and physical coordinates ``X``, ``Y``, ``Z``.
    components : {"cartesian", "contravariant"}, optional
        How to read the components: Cartesian x/y/z, or contravariant logical components,
        pushed forward by :func:`push_forward`. Default: ``"cartesian"``.
    n_points : int, optional
        The number of seed points (at most the number of grid points when seeding on the grid).
        Default: 100.
    source_radius : float, optional
        Seed in a sphere of this radius instead of at grid points. Default (when
        ``source_center`` is given): a quarter of the domain size.
    source_center : (float, float, float), optional
        Seed in a sphere around this physical point instead of at grid points. Default (when
        ``source_radius`` is given): the bounding-box center.
    max_length : float, optional
        Maximum length of each line. Default: four times the domain size.
    tube_radius : float, optional
        Draw the lines as tubes of this radius. Default: plain lines.
    cmap : str or matplotlib colormap, optional
        The colormap of the magnitude. Default: ``"viridis"``.
    show_domain : bool, optional
        Draw the domain's outer surface translucently for context. Default: ``True``.
    title : str, optional
        Text in the scene's corner. Default: ``"<label> field lines"``; ``""`` for none.
    plotter : pyvista.Plotter, optional
        Draw into this scene instead of a new one, to combine several views. The camera is only
        aimed at the field when this function creates the plotter.

    Returns
    -------
    pyvista.Plotter
        The scene, not yet shown: call ``.show()`` or ``.screenshot(path)``.

    Raises
    ------
    ValueError
        If ``components`` is unknown, or the field isn't a three-component field over the
        logical dimensions with physical coordinates.

    See Also
    --------
    pyvista_glyphs : Arrows of the vector field.

    Examples
    --------
    >>> pyvista_streamlines(B.isel(t=-1)).show()
    >>> pyvista_streamlines(
    ...     B.isel(t=-1),
    ...     source_center=(3.0, 0.0, 0.0),
    ...     source_radius=0.5,
    ...     tube_radius=0.01,
    ... ).show()
    """
    pv = _pv()
    name = _label(data)
    grid = _vector_grid(data, components, name)
    max_length = 4 * grid.length if max_length is None else max_length
    # Field lines depend only on the direction: trace the unit field, which keeps VTK's
    # adaptive integrator independent of the field's magnitude (Struphy perturbations are often
    # ~1e-5), with small steps for thin curved cells; lines are colored by the real magnitude.
    vectors = np.asarray(grid[name])
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    grid["_direction"] = np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm > 0)
    tracing = dict(
        vectors="_direction",
        max_length=max_length,
        integration_direction="both",
        surface_streamlines=is_flat(grid),
        step_unit="cl",
        initial_step_length=0.1,
        max_steps=4000,
    )
    if source_center is None and source_radius is None:
        # Seed at grid points: a sphere around the bounding-box centre can lie outside a
        # curved domain (e.g. a torus sector), and would miss a 2-D plane entirely.
        rng = np.random.default_rng(0)
        seeds = rng.choice(grid.n_points, size=min(n_points, grid.n_points), replace=False)
        lines = grid.streamlines_from_source(pv.PolyData(np.asarray(grid.points)[np.sort(seeds)]), **tracing)
    else:
        lines = grid.streamlines(
            n_points=n_points,
            source_radius=(0.25 * grid.length if source_radius is None else source_radius),
            source_center=grid.center if source_center is None else source_center,
            **tracing,
        )
    own = plotter is None
    plotter = _plotter(plotter)
    _add_context(plotter, grid, show_domain)
    if lines.n_points:
        if tube_radius:
            lines = lines.tube(radius=tube_radius)
        plotter.add_mesh(
            lines,
            scalars=f"|{name}|",
            cmap=cmap,
            line_width=2,
            scalar_bar_args=_bar(f"|{name}|"),
            name="streamlines",
        )
    return _finish(
        plotter,
        f"{name} field lines" if title is None else title,
        grid if own else None,
    )


def orbit_polylines(orbits: xr.Dataset, *, color_by: str = "t", max_markers: int = 200):
    """Marker orbits as one ``pyvista.PolyData`` line per marker, with point data ``color_by``.

    Samples where a marker is lost (every quantity zero) are dropped. ``color_by`` is ``"t"``,
    ``"classification"`` (see :func:`~plasma_plots.analysis.classify_orbits`), or the name of
    any ``(t, marker)`` variable, e.g. ``"v_par"`` or ``"weight"``.

    Parameters
    ----------
    orbits : xarray.Dataset
        Marker orbits with dims ``(t, marker)`` and physical positions ``x``, ``y``, ``z``.
    color_by : str, optional
        The point data to attach: ``"t"``, ``"classification"`` or a variable name.
        Default: ``"t"``.
    max_markers : int, optional
        At most this many markers are used. Default: 200.

    Returns
    -------
    pyvista.PolyData
        One line per marker with at least two samples left, and point data ``color_by``;
        empty if no marker has.

    Raises
    ------
    ValueError
        If ``color_by`` is neither ``"t"``, ``"classification"`` nor a variable of ``orbits``.
    """
    pv = _pv()
    from .plotting import prepare_orbits

    subset = prepare_orbits(orbits, max_markers=max_markers, required=("x", "y", "z"))
    subset = subset.transpose("t", "marker", ...)
    positions = np.stack([np.asarray(subset[c]) for c in ("x", "y", "z")], axis=-1)
    alive = np.ones(positions.shape[:2], dtype=bool)
    for name in subset.data_vars:
        if subset[name].dims == ("t", "marker"):
            alive &= np.asarray(subset[name]) == 0
    alive = ~alive
    if color_by == "t":
        colors = np.broadcast_to(np.asarray(subset.t)[:, None], alive.shape)
    elif color_by == "classification":
        colors = np.broadcast_to(np.asarray(classify_orbits(subset))[None, :], alive.shape)
    elif color_by in subset.data_vars:
        colors = np.asarray(subset[color_by].transpose("t", "marker"))
    else:
        raise ValueError(f'color_by must be "t", "classification" or a variable of {tuple(subset.data_vars)}')
    points, cells, scalars = [], [], []
    for marker in range(positions.shape[1]):
        keep = np.flatnonzero(alive[:, marker])
        if keep.size < 2:
            continue
        start = sum(len(p) for p in points)
        points.append(positions[keep, marker])
        scalars.append(colors[keep, marker])
        cells.append(np.concatenate([[keep.size], start + np.arange(keep.size)]))
    if not points:
        return pv.PolyData()
    lines = pv.PolyData(np.concatenate(points), lines=np.concatenate(cells))
    lines.point_data[color_by] = np.concatenate(scalars).astype(float)
    return lines


@rank_zero
def pyvista_orbits(
    orbits: xr.Dataset,
    *,
    color_by: str = "t",
    max_markers: int = 200,
    tube_radius: float | None = None,
    cmap=None,
    domain: xr.DataArray | None = None,
    title: str | None = None,
    plotter=None,
):
    """Marker orbits as 3-D lines (or tubes), colored by time, orbit class, or any variable.

    ``domain`` is any field with physical coordinates whose outer surface is drawn translucently
    for context. See :func:`orbit_polylines` for ``color_by``.

    Parameters
    ----------
    orbits : xarray.Dataset
        Marker orbits with dims ``(t, marker)`` and physical positions ``x``, ``y``, ``z``.
    color_by : str, optional
        ``"t"``, ``"classification"`` (passing, trapped, lost, with a legend) or the name of
        any ``(t, marker)`` variable, e.g. ``"v_par"``. Default: ``"t"``.
    max_markers : int, optional
        At most this many markers are drawn. Default: 200.
    tube_radius : float, optional
        Draw the orbits as tubes of this radius. Default: plain lines.
    cmap : str or matplotlib colormap, optional
        The colormap (not used for ``"classification"``). Default: ``"viridis"``.
    domain : xarray.DataArray, optional
        A field with physical coordinates whose outer surface is drawn translucently; other
        than ``eta1``, ``eta2``, ``eta3``, its dimensions are taken at their first position.
    title : str, optional
        Text in the scene's corner. Default: ``"Marker orbits"``; ``""`` for none.
    plotter : pyvista.Plotter, optional
        Draw into this scene instead of a new one, to combine several views.

    Returns
    -------
    pyvista.Plotter
        The scene, not yet shown: call ``.show()`` or ``.screenshot(path)``.

    Raises
    ------
    ValueError
        If ``color_by`` is neither ``"t"``, ``"classification"`` nor a variable of ``orbits``.

    See Also
    --------
    orbit_polylines : The orbits as a ``pyvista.PolyData``, without drawing them.

    Examples
    --------
    >>> pyvista_orbits(out.kinetic_ions.orbits, color_by="classification").show()
    >>> pyvista_orbits(
    ...     out.kinetic_ions.orbits, color_by="v_par", domain=phi, tube_radius=0.01
    ... ).show()
    """
    lines = orbit_polylines(orbits, color_by=color_by, max_markers=max_markers)
    plotter = _plotter(plotter)
    if domain is not None:
        spatial = domain.isel({d: 0 for d in domain.dims if d not in logical_dims(domain)})
        _add_context(plotter, structured_grid(spatial), True)
    if lines.n_points and color_by == "classification":
        legend = []
        for code, name in ORBIT_CLASSES.items():
            part = lines.threshold((code - 0.5, code + 0.5), scalars=color_by)
            if not part.n_points:
                continue
            part = part.extract_surface(algorithm="dataset_surface")
            mesh = part.tube(radius=tube_radius) if tube_radius else part
            color = ORBIT_CLASS_COLORS[name]
            plotter.add_mesh(mesh, color=color, line_width=2, name=f"orbits_{name}")
            legend.append([name, color])
        plotter.add_legend(legend, bcolor=None, size=(0.15, 0.12), face="line")
    elif lines.n_points:
        mesh = lines.tube(radius=tube_radius) if tube_radius else lines
        plotter.add_mesh(
            mesh,
            scalars=color_by,
            cmap=cmap or "viridis",
            line_width=2,
            scalar_bar_args=_bar(color_by),
            name="orbits",
        )
    return _finish(plotter, "Marker orbits" if title is None else title)


@rank_zero
def pyvista_domain(
    domain,
    *,
    n1: int = 8,
    n2: int = 32,
    n3: int = 32,
    resolution: int = 4,
    color="black",
    surface: bool = True,
    cross_section: bool = True,
    title: str | None = None,
    plotter=None,
):
    """The mapping of a Struphy ``domain`` as a wireframe of logical grid lines.

    Grid lines are drawn on the real boundary faces (see :func:`boundary_keys`; for a torus the
    outer and inner surfaces) and, with ``cross_section``, over the whole ``eta3 = 0`` face (for a
    torus a poloidal cross-section). ``n1``, ``n2``, ``n3`` lines per direction, each sampled
    ``resolution`` times finer so curved lines stay smooth. ``surface`` adds the translucent
    boundary. Useful to check the geometry (and its orientation) of a run; ``n3=1`` shows a
    2-D run's plane.

    Parameters
    ----------
    domain : callable
        A Struphy domain (mapping), e.g. ``out.domain``, called as
        ``domain(eta1, eta2, eta3, squeeze_out=False)`` to give ``x``, ``y``, ``z``.
    n1 : int, optional
        Grid lines along ``eta1``. Default: 8.
    n2 : int, optional
        Grid lines along ``eta2``. Default: 32.
    n3 : int, optional
        Grid lines along ``eta3``. Default: 32.
    resolution : int, optional
        Samples per line spacing, so curved lines stay smooth. Default: 4.
    color : str, optional
        Color of the grid lines. Default: ``"black"``.
    surface : bool, optional
        Draw the translucent boundary surface. Default: ``True``.
    cross_section : bool, optional
        Also draw grid lines over the ``eta3 = 0`` face. Default: ``True``.
    title : str, optional
        Text in the scene's corner. Default: ``"Domain"``; ``""`` for none.
    plotter : pyvista.Plotter, optional
        Draw into this scene instead of a new one, to combine several views. The camera is only
        aimed at the domain when this function creates the plotter.

    Returns
    -------
    pyvista.Plotter
        The scene, not yet shown: call ``.show()`` or ``.screenshot(path)``.

    Examples
    --------
    >>> pyvista_domain(out.domain).show()
    >>> pyvista_domain(out.domain, n3=1, surface=False).show()
    """
    pv = _pv()
    fine = [np.linspace(0.0, 1.0, max((n - 1) * resolution + 1, 1)) for n in (n1, n2, n3)]
    x, y, z = (np.asarray(c, dtype=float) for c in domain(*fine, squeeze_out=False))
    grid = pv.StructuredGrid(x, y, z)
    own = plotter is None
    plotter = _plotter(plotter)
    if surface:
        _add_context(plotter, grid, True)
    points = np.stack([x, y, z], axis=-1)
    keys = boundary_keys(points)
    if cross_section and (2, 0) not in keys and points.shape[2] > 1:
        keys.append((2, 0))
    segments, cells = [], []
    for axis, index in keys:
        for line in _face_lines(_face(points, axis, index), resolution):
            start = sum(len(s) for s in segments)
            segments.append(line)
            cells.append(np.concatenate([[len(line)], start + np.arange(len(line))]))
    if segments:
        wires = pv.PolyData(np.concatenate(segments), lines=np.concatenate(cells))
        plotter.add_mesh(wires, color=color, line_width=1, name="wireframe")
    return _finish(plotter, "Domain" if title is None else title, grid if own else None)


def _surface(mesh):
    """A ``pyvista.PolyData`` of a mesh: any pyvista dataset, a ``(vertices, faces)`` pair, or
    an object with ``vertices`` and ``cells`` (or ``faces``), such as simgeometry's ``SurfaceMesh``."""
    pv = _pv()
    if isinstance(mesh, pv.PolyData):
        return mesh
    if isinstance(mesh, pv.DataSet):
        return mesh.extract_surface(algorithm="dataset_surface")
    if _is_vertices_faces(mesh):
        vertices, cells = mesh
    elif hasattr(mesh, "vertices") and (hasattr(mesh, "cells") or hasattr(mesh, "faces")):
        vertices, cells = mesh.vertices, getattr(mesh, "cells", None)
        cells = mesh.faces if cells is None else cells
    else:
        raise TypeError(
            "a mesh is a pyvista dataset, a (vertices, faces) pair of arrays, or has vertices and "
            f"cells; got {type(mesh).__name__}"
        )
    vertices = np.asarray(vertices, dtype=float)
    cells = np.asarray(cells, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] not in (2, 3):
        raise ValueError(f"vertices must have shape (n, 3) (or (n, 2) in a plane); got {vertices.shape}")
    if vertices.shape[1] == 2:  # a 2-D outline in the z = 0 plane
        vertices = np.column_stack([vertices, np.zeros(len(vertices))])
    if cells.ndim != 2 or cells.shape[1] < 2:
        raise ValueError(f"faces must have shape (m, 3) (or (m, k) polygons, (m, 2) segments); got {cells.shape}")
    if len(cells) and (cells.min() < 0 or cells.max() >= len(vertices)):
        raise ValueError(f"faces index {len(vertices)} vertices; got indices {cells.min()} to {cells.max()}")
    packed = np.column_stack([np.full(len(cells), cells.shape[1]), cells]).ravel()
    if cells.shape[1] == 2:
        return pv.PolyData(vertices, lines=packed)
    return pv.PolyData(vertices, faces=packed)


def _is_vertices_faces(mesh) -> bool:
    if not isinstance(mesh, (tuple, list)) or len(mesh) != 2:
        return False
    try:
        arrays = [np.asarray(part) for part in mesh]
    except (TypeError, ValueError):  # ragged: two meshes, not one pair
        return False
    return all(array.ndim == 2 and array.dtype.kind in "iuf" for array in arrays)


def _per_mesh(value, labels, name):
    """One value per mesh from a single value, a sequence, or a dict keyed by label."""
    if isinstance(value, dict):
        missing = [label for label in labels if label not in value]
        if missing:
            raise ValueError(f"{name} has no entry for {missing}")
        return [value[label] for label in labels]
    if isinstance(value, (list, tuple)) and not (name == "colors" and _is_color(value)):
        if len(value) != len(labels):
            raise ValueError(f"{name} needs one entry per mesh ({len(labels)}); got {len(value)}")
        return list(value)
    return [value] * len(labels)


def _is_color(value) -> bool:
    """Whether ``value`` is one RGB(A) color such as ``(0.2, 0.4, 0.6)``, not a list of colors."""
    return len(value) in (3, 4) and all(isinstance(v, (int, float, np.number)) for v in value)


@rank_zero
def pyvista_surfaces(
    meshes,
    *,
    labels=None,
    colors=None,
    scalars=None,
    cmap=None,
    clim=None,
    opacity=1.0,
    show_edges: bool = False,
    title: str | None = None,
    legend: bool = True,
    plotter=None,
):
    """Surface meshes in physical space, such as geometry or marching-cubes surfaces.

    Each mesh gets its own color from Matplotlib's color cycle (``C0``, ``C1``, ...), and the
    labels go in a legend when there is more than one. With ``scalars``, the surfaces are
    colored by a value instead, on one color scale with a scalar bar, e.g. conductors by their
    potential. A flat scene (every point in one plane) is viewed face on.

    Parameters
    ----------
    meshes : mesh, sequence of meshes, or dict of str to mesh
        A mesh is a ``pyvista.PolyData`` (any other pyvista dataset is drawn by its surface), a
        ``(vertices, faces)`` pair of arrays with shapes ``(n, 3)`` and ``(m, 3)`` (triangles,
        as from ``skimage.measure.marching_cubes``), or an object with ``vertices`` and
        ``cells`` such as simgeometry's ``SurfaceMesh``. Faces with two columns are line
        segments, and ``(n, 2)`` vertices lie in the ``z = 0`` plane. A dict's keys are the
        labels. Empty meshes are skipped.
    labels : sequence of str, optional
        One label per mesh, for the legend. Default: the dict's keys, else none.
    colors : color, sequence of colors, or dict of str to color, optional
        Colors of the meshes, by position or by label; any Matplotlib color. Default:
        Matplotlib's color cycle.
    scalars : str or callable, optional
        Color by this point-data array of every mesh, or by ``scalars(points)``, a function of
        the ``(n, 3)`` points that returns ``n`` values. The scalar bar shows the array's name,
        or the function's name. Default: plain colors.
    cmap : str or matplotlib colormap, optional
        The colormap for ``scalars``. Default: ``"viridis"``.
    clim : (float, float), optional
        Color limits for ``scalars``. Default: the range over all meshes.
    opacity : float, sequence of float, or dict of str to float, optional
        Opacity of the meshes, one for all or by position or label. Default: 1.
    show_edges : bool, optional
        Draw the mesh edges. Default: ``False``.
    title : str, optional
        Text in the scene's corner. Default: none.
    legend : bool, optional
        Show a legend of the labels when there are two or more (not with ``scalars``).
        Default: ``True``.
    plotter : pyvista.Plotter, optional
        Draw into this scene instead of a new one, to combine several views. The camera is only
        aimed at the surfaces when this function creates the plotter.

    Returns
    -------
    pyvista.Plotter
        The scene, not yet shown: call ``.show()`` or ``.screenshot(path)``.

    Raises
    ------
    TypeError
        If a mesh is none of the accepted kinds.
    ValueError
        If ``labels``, ``colors`` or ``opacity`` don't match the meshes, faces index missing
        vertices, or a mesh lacks the ``scalars`` array.

    Examples
    --------
    >>> import pyvista as pv
    >>> pyvista_surfaces(
    ...     {"inner": pv.Sphere(radius=0.5), "outer": pv.Sphere()},
    ...     opacity={"inner": 1.0, "outer": 0.3},
    ... ).show()
    >>> vertices, faces, _, _ = skimage.measure.marching_cubes(values, level=0)
    >>> pyvista_surfaces(
    ...     (vertices, faces), scalars=lambda p: p[:, 2], show_edges=True
    ... ).show()
    """
    from matplotlib.colors import to_hex

    if isinstance(meshes, dict):
        if labels is not None:
            raise ValueError("pass labels either as the dict's keys or as labels, not both")
        labels, meshes = [str(label) for label in meshes], list(meshes.values())
    elif _is_vertices_faces(meshes) or not isinstance(meshes, (list, tuple)):
        meshes = [meshes]
    meshes = [_surface(mesh) for mesh in meshes]
    named = labels is not None
    labels = list(labels) if named else [f"surface {i}" for i in range(len(meshes))]
    if len(labels) != len(meshes):
        raise ValueError(f"labels needs one entry per mesh ({len(meshes)}); got {len(labels)}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"labels must be distinct; got {labels}")
    colors = [f"C{i}" for i in range(len(meshes))] if colors is None else _per_mesh(colors, labels, "colors")
    colors = [to_hex(color) for color in colors]
    opacities = [float(value) for value in _per_mesh(opacity, labels, "opacity")]

    bar = None
    if scalars is not None:
        if callable(scalars):
            name = getattr(scalars, "__name__", "<lambda>")
            bar = "value" if name.startswith("<") else name
            computed = []
            for mesh in meshes:
                mesh = mesh.copy(deep=False)  # leave the caller's mesh as it is
                if mesh.n_points:
                    mesh.point_data[bar] = np.asarray(scalars(np.asarray(mesh.points)), dtype=float)
                computed.append(mesh)
            meshes = computed
        else:
            bar = scalars
            missing = [label for label, mesh in zip(labels, meshes) if mesh.n_points and bar not in mesh.point_data]
            if missing:
                raise ValueError(f"no point data {bar!r} on {missing}")
        # NaN marks points without a value (drawn in the NaN color), e.g. objects that are not
        # conductors; a mesh may be NaN throughout.
        values = np.concatenate([np.asarray(mesh.point_data[bar], dtype=float).ravel() for mesh in meshes] or [[]])
        values = values[np.isfinite(values)]
        if clim is None and values.size:
            clim = (float(values.min()), float(values.max()))

    own = plotter is None
    plotter = _plotter(plotter)
    drawn = []
    for label, mesh, color, alpha in zip(labels, meshes, colors, opacities):
        if not mesh.n_points:
            continue
        options = dict(opacity=alpha, show_edges=show_edges, line_width=2)
        if bar is None:
            options["color"] = color
        else:
            options.update(scalars=bar, cmap=cmap or "viridis", clim=clim, scalar_bar_args=_bar(bar))
        plotter.add_mesh(mesh, **options)
        drawn.append((label, color, mesh))
    if legend and named and bar is None and len(drawn) > 1:
        plotter.add_legend([[label, color] for label, color, _ in drawn], bcolor=None, face="rectangle")
    if own and drawn:
        _camera(plotter, _pv().merge([mesh for _, _, mesh in drawn]))
    return _finish(plotter, title)


@rank_zero
def save_vtk(data: xr.DataArray, path, *, name: str | None = None) -> list[str]:
    """Write a field to VTK structured grids (``.vts``) on its physical points, for ParaView.

    With a ``t`` dimension, one file per time is written into the directory ``path``, plus a
    ``.pvd`` collection that ParaView opens as a time series; without one, ``path`` is a single
    ``.vts`` file. Vector fields (``component``) become point vectors. Any array works, e.g. a
    :func:`~plasma_plots.spectral.filter_time` result, so filtered modes can be inspected in
    ParaView too.

    Parameters
    ----------
    data : xarray.DataArray
        A field over ``(eta1, eta2, eta3)`` (and optionally ``t`` and ``component``) with
        physical coordinates ``X``, ``Y``, ``Z``; every other dimension selected.
    path : str or pathlib.Path
        The ``.vts`` file (the suffix is set to ``.vts``), or with a ``t`` dimension the
        directory to write into (created if needed).
    name : str, optional
        The name of the point data, also the file stem in a time series. Default: the field's
        label.

    Returns
    -------
    list of str
        The written paths: the ``.vts`` file, or the ``.pvd`` collection followed by one
        ``.vts`` file per time.

    Examples
    --------
    >>> save_vtk(phi, "vtk/phi")
    >>> save_vtk(phi.isel(t=-1), "phi_last.vts")
    """
    from pathlib import Path as _Path

    path = _Path(path)
    name = name or _label(data)
    if "t" not in data.dims:
        path.parent.mkdir(parents=True, exist_ok=True)
        target = path if path.suffix == ".vts" else path.with_suffix(".vts")
        structured_grid(data, name=name).save(str(target))
        return [str(target)]
    path.mkdir(parents=True, exist_ok=True)
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in name) or "field"
    written, entries = [], []
    for index in range(data.sizes["t"]):
        target = path / f"{stem}_{index:04d}.vts"
        structured_grid(data.isel(t=index), name=name).save(str(target))
        written.append(str(target))
        entries.append(f'    <DataSet timestep="{float(data.t[index])!r}" file="{target.name}"/>')
    collection = path / f"{stem}.pvd"
    collection.write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n'
        + "\n".join(entries)
        + "\n  </Collection>\n</VTKFile>\n"
    )
    return [str(collection), *written]


RENDERERS = {
    "isosurface": pyvista_isosurface,
    "slices": pyvista_slices,
    "glyphs": pyvista_glyphs,
    "streamlines": pyvista_streamlines,
}


@rank_zero
def save_movie(
    data: xr.DataArray,
    path,
    *,
    kind: Literal["isosurface", "slices", "glyphs", "streamlines"] = "slices",
    sweep: str = "t",
    step: int = 1,
    framerate: int = 10,
    clim=None,
    window_size=(1024, 768),
    **options,
):
    """Render one 3-D view per ``sweep`` step into a GIF (``.gif``) or video (``.mp4``, ...).

    ``kind`` picks the view and ``options`` are passed on to it. Scalar views share color
    limits over the whole sweep (override with ``clim``), and the camera is fixed after the
    first frame. GIFs need ``imageio``, videos ``imageio-ffmpeg``.

    Parameters
    ----------
    data : xarray.DataArray
        A field with dimension ``sweep`` plus what the view ``kind`` needs (see
        :func:`pyvista_slices`, :func:`pyvista_glyphs`, ...).
    path : str or pathlib.Path
        The output file: a GIF for ``.gif``, a video (e.g. ``.mp4``) for any other suffix.
    kind : {"isosurface", "slices", "glyphs", "streamlines"}, optional
        The view of each frame. Default: ``"slices"``.
    sweep : str, optional
        The dimension to step through, one frame per step. Default: ``"t"``.
    step : int, optional
        Use every ``step``-th position of ``sweep``. Default: 1.
    framerate : int, optional
        Frames per second. Default: 10.
    clim : (float, float), optional
        Color limits of the scalar views (``"isosurface"``, ``"slices"``). Default: from the
        whole sweep, with the ``symmetric`` and ``robust`` options.
    window_size : (int, int), optional
        Frame size in pixels. Default: ``(1024, 768)``.
    **options
        Passed on to the view, e.g. ``cuts`` or ``cmap``. A ``title`` is prefixed to each
        frame's ``"<sweep> = <value>"`` label (default: the field's label).

    Returns
    -------
    str
        The path of the written file.

    Raises
    ------
    ValueError
        If ``kind`` is unknown, ``step`` isn't a positive integer, or ``data`` has no ``sweep``
        dimension.

    Examples
    --------
    >>> save_movie(phi, "phi.gif", cuts={"eta3": 0}, symmetric=True)
    >>> save_movie(B, "B.mp4", kind="glyphs", step=2)
    """
    pv = _pv()
    if kind not in RENDERERS:
        raise ValueError(f"kind must be one of {tuple(RENDERERS)}; got {kind!r}")
    if not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("step must be a positive integer")
    validate_array(data, required_dims=(sweep,))
    render = RENDERERS[kind]
    if kind in ("isosurface", "slices"):
        options["clim"] = _clim(
            np.asarray(data),
            clim,
            symmetric=options.pop("symmetric", False),
            robust=options.pop("robust", False),
        )
    path = Path(path)
    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    if path.suffix.lower() == ".gif":
        plotter.open_gif(str(path), fps=framerate)
    else:
        plotter.open_movie(str(path), framerate=framerate)
    title = options.pop("title", None) or _label(data)
    try:
        for frame, index in enumerate(range(0, data.sizes[sweep], step)):
            snapshot = data.isel({sweep: index})
            label = f"{title} at {sweep} = {float(data[sweep][index]):.3e}"
            if frame:
                plotter.clear_actors()
            render(snapshot, plotter=plotter, title=label, **options)
            if not frame:
                grid = structured_grid(snapshot)
                if is_flat(grid):
                    _camera(plotter, grid)
                else:
                    plotter.camera_position = "iso"
                    plotter.reset_camera()
            plotter.write_frame()
    finally:
        plotter.close()
    return str(path)


SCENE_EXTRA = 'pip install "plasma-plots[scene]"'


class Scene:
    """A ``pyvista.Plotter`` displayed as an interactive 3-D scene; see :func:`scene`.

    The PNG and the HTML are rendered from the plotter the first time they are needed and
    kept, so later changes to the plotter don't show.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The scene, off screen.
    alt : str, optional
        A text alternative for the screenshot.
    title : str, optional
        A title for the scene.
    window_size : (int, int), optional
        The screenshot's size in pixels. Default: the plotter's window size.
    """

    def __init__(self, plotter, *, alt: str | None = None, title: str | None = None, window_size=None):
        self.plotter = plotter
        self.alt = alt
        self.title = title
        self.window_size = window_size
        self._png: bytes | None = None
        self._html: str | None = None

    def png(self) -> bytes:
        """The screenshot of the plotter, as PNG bytes.

        Returns
        -------
        bytes
            The PNG file's content.

        Raises
        ------
        RuntimeError
            If the plotter isn't off screen and hasn't been shown.
        """
        if self._png is None:
            buffer = io.BytesIO()
            try:
                self.plotter.screenshot(buffer, return_img=False, window_size=self.window_size)
            except RuntimeError as error:
                raise RuntimeError(
                    "scene() takes a screenshot, which needs an off-screen plotter: "
                    "pyvista.Plotter(off_screen=True), or pyvista.OFF_SCREEN = True before creating it"
                ) from error
            self._png = buffer.getvalue()
        return self._png

    def html(self) -> str:
        """The scene as a standalone HTML page, vtk.js inlined (``plotter.trame.export_html``).

        Scalar bars are hidden in the export: vtk.js receives colors already mapped to RGB, so
        its color bar would read 0-255. The PNG keeps the correct one.

        Returns
        -------
        str
            The HTML document.

        Raises
        ------
        ImportError
            If trame-pyvista isn't installed (the ``scene`` extra).
        """
        if self._html is None:
            try:
                trame = self.plotter.trame
            except AttributeError as error:
                raise ImportError(f"Interactive scenes need the optional extra: {SCENE_EXTRA}") from error
            bars = [actor for actor in self.plotter.scalar_bars.values() if actor.GetVisibility()]
            for actor in bars:
                actor.VisibilityOff()
            try:
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / "scene.html"
                    trame.export_html(str(path))
                    self._html = path.read_text(encoding="utf-8")
            except ImportError as error:  # e.g. nest-asyncio2, to start trame's server
                raise ImportError(f"Interactive scenes need the optional extra: {SCENE_EXTRA} ({error})") from error
            finally:
                for actor in bars:
                    actor.VisibilityOn()
        return self._html

    def save_png(self, path) -> str:
        """Write the screenshot to a PNG file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write.

        Returns
        -------
        str
            ``path``, as a string.
        """
        Path(path).write_bytes(self.png())
        return str(path)

    def save_html(self, path) -> str:
        """Write the scene to a standalone HTML file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file to write.

        Returns
        -------
        str
            ``path``, as a string.
        """
        Path(path).write_text(self.html(), encoding="utf-8")
        return str(path)

    def _repr_mimebundle_(self, include=None, exclude=None):
        # the screenshot first: it renders the plotter, which the export then reuses
        png = base64.b64encode(self.png()).decode("ascii")
        data = {"text/html": self.html(), "image/png": png}
        info = {key: value for key, value in (("alt", self.alt), ("title", self.title)) if value is not None}
        return data, {"pyvista-scene": info}


def scene(plotter, *, alt: str | None = None, title: str | None = None, window_size=None) -> Scene:
    """Show a ``pyvista.Plotter`` as an interactive 3-D scene in a notebook.

    As the last expression of a cell, the result displays as a standalone vtk.js page
    (``text/html``) plus a screenshot (``image/png``). Jupyter shows the page; docs sites built
    from the notebook show the screenshot and swap in the scene on click. ``alt`` and ``title``
    travel in the output's ``"pyvista-scene"`` metadata.

    The screenshot needs an off-screen plotter: plotters created in a notebook are, and in a
    script set ``pyvista.OFF_SCREEN = True`` (or pass ``off_screen=True``) before creating
    one. The export needs trame-pyvista: ``pip install "plasma-plots[scene]"``.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The scene, e.g. from ``phi.plasma.plot.isosurface(...)``. Set its camera first.
    alt : str, optional
        A text alternative for the screenshot.
    title : str, optional
        A title for the scene, e.g. of the frame it is shown in.
    window_size : (int, int), optional
        The screenshot's size in pixels. Default: the plotter's window size.

    Returns
    -------
    Scene
        Displays itself in Jupyter; ``.save_png(path)`` and ``.save_html(path)`` write the two
        files, e.g. for a docs page.

    Examples
    --------
    >>> plotter = phi.plasma.plot.isosurface(values=[-0.5, 0.5], t=-1)
    >>> plasma_plots.scene(plotter, alt="Two isosurfaces of phi")
    """
    return Scene(plotter, alt=alt, title=title, window_size=window_size)
