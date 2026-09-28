"""Three-dimensional PyVista views of labeled Struphy output.

Every function returns a ``pyvista.Plotter`` without showing it: call ``.show()`` in an
interactive session or notebook, or ``.screenshot(path)`` (with ``pyvista.OFF_SCREEN = True``
in batch jobs). Pass ``plotter=`` to draw several views into one scene.

Fields need their physical ``X``, ``Y``, ``Z`` coordinates attached, as every Struphy field
product has; orbits need physical positions ``x``, ``y``, ``z``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from .analysis import ORBIT_CLASSES, classify_orbits
from .arrays import close_periodic, mapping_jacobian, validate_array, value_label
from .mpi import rank_zero

SPATIAL = ("eta1", "eta2", "eta3")
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
    others = set(data.dims) - {*extra, *SPATIAL}
    if others:
        raise ValueError(f"select every dimension except {(*extra, *SPATIAL)} first; {sorted(others)} remain")
    missing = [name for name in ("X", "Y", "Z") if name not in data.coords]
    if missing:
        raise ValueError(f"3-D views need physical coordinates X, Y, Z on {data.name!r}; missing {missing}")
    absent = [dim for dim in SPATIAL if dim not in data.dims]
    if len(absent) > 1:
        raise ValueError(f"3-D views need at least two of eta1, eta2, eta3; {data.name!r} has dims {data.dims}")
    for dim in absent:
        data = data.expand_dims(dim) if dim in data.coords else data.expand_dims({dim: [0.0]})
    coords = {}
    for name in ("X", "Y", "Z"):
        coordinate = data.coords[name]
        for dim in SPATIAL:
            if dim not in coordinate.dims:
                coordinate = coordinate.expand_dims({dim: data.sizes[dim]})
        coords[name] = coordinate.transpose(*SPATIAL).variable
    return data.assign_coords(coords).transpose(*extra, *SPATIAL)


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
    """Look straight at a planar grid (a 2-D run); leave 3-D scenes at the default view."""
    if not is_flat(grid):
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
    return tuple(np.asarray(data.coords[name].transpose(*SPATIAL), dtype=float) for name in ("X", "Y", "Z"))


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
    data = close_periodic(_spatial(data, extra=("component",) if vector else ()), SPATIAL)
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
    >>> B_xyz = push_forward(out.evaluate("em_fields/b_field", representation="v").isel(t=-1))
    """
    data = _spatial(data, extra=("component",))
    if data.sizes["component"] != 3:
        raise ValueError(f"a vector field needs 3 components; got {data.sizes['component']}")
    short = [dim for dim in SPATIAL if data.sizes[dim] < 2]
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


def _bar(title):
    return {"title": title, "fmt": "%.3g"}


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


def _add_context(plotter, grid, show_domain):
    if not show_domain:
        return
    pv = _pv()
    for i, face in enumerate(boundary_faces(_grid_points(grid))):
        surface = pv.StructuredGrid(face[..., 0], face[..., 1], face[..., 2])
        plotter.add_mesh(surface, color="lightgrey", opacity=0.12, name=f"domain{i}")


def _finish(plotter, title, grid=None):
    """Title and axes; with ``grid`` (a plotter this module created), aim the camera at it."""
    if title:
        plotter.add_text(title, font_size=10, name="title")
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
        if dim not in SPATIAL:
            raise ValueError(f"cuts are along eta1, eta2 or eta3; got {dim!r}")
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
        cuts = {dim: data.sizes[dim] // 2 for dim in SPATIAL}
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
    if not (is_flat(grid) and len(pieces) == 1 and pieces[0].shape == data.shape):
        _add_context(plotter, grid, show_domain)
    for i, piece in enumerate(pieces):
        plotter.add_mesh(
            structured_grid(piece, name=_label(data)),
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
    thinned = _vector_grid(data.isel({dim: slice(None, None, stride) for dim in SPATIAL}), components, name)
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
    >>> pyvista_streamlines(B.isel(t=-1), source_center=(3.0, 0.0, 0.0), source_radius=0.5, tube_radius=0.01).show()
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
    >>> pyvista_orbits(out.kinetic_ions.orbits, color_by="v_par", domain=phi, tube_radius=0.01).show()
    """
    lines = orbit_polylines(orbits, color_by=color_by, max_markers=max_markers)
    plotter = _plotter(plotter)
    if domain is not None:
        spatial = domain.isel({d: 0 for d in domain.dims if d not in SPATIAL})
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
