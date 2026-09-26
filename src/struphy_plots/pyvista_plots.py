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
from .arrays import validate_array, value_label

SPATIAL = ("e1", "e2", "e3")
ORBIT_CLASS_COLORS = {"passing": "tab:blue", "trapped": "tab:orange", "lost": "grey"}


def _pv():
    try:
        import pyvista
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            'PyVista plots need the optional extra: pip install "struphy-plots[pyvista]"'
        ) from error
    return pyvista


def _label(data):
    return data.attrs.get("label") or data.attrs.get("long_name") or data.name or "value"


def _plotter(plotter):
    return _pv().Plotter() if plotter is None else plotter


def _spatial(data: xr.DataArray, *, extra=()) -> xr.DataArray:
    """``data`` transposed to ``(*extra, e1, e2, e3)``, checking nothing else is left."""
    validate_array(data, required_dims=(*extra, *SPATIAL))
    others = set(data.dims) - {*extra, *SPATIAL}
    if others:
        raise ValueError(
            f"select every dimension except {(*extra, *SPATIAL)} first; {sorted(others)} remain"
        )
    missing = [name for name in ("X", "Y", "Z") if name not in data.coords]
    if missing:
        raise ValueError(
            f"3-D views need physical coordinates X, Y, Z on {data.name!r}; missing {missing}"
        )
    return data.transpose(*extra, *SPATIAL)


def _points(data: xr.DataArray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.asarray(data.coords[name].transpose(*SPATIAL), dtype=float) for name in ("X", "Y", "Z")
    )


def structured_grid(data: xr.DataArray, *, name: str | None = None):
    """A ``pyvista.StructuredGrid`` of a selected ``(e1, e2, e3)`` field on its physical points.

    A scalar field becomes point data ``name`` (default: the field's label); a vector field with
    a ``component`` dimension of three Cartesian components becomes point vectors ``name``, plus
    their magnitude as ``"|name|"``. The grid is useful directly for any PyVista filter.
    """
    pv = _pv()
    vector = "component" in data.dims
    data = _spatial(data, extra=("component",) if vector else ())
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

    ``data`` has dims ``(component, e1, e2, e3)``, e.g. from
    ``out.evaluate(name, eta1=..., eta2=..., eta3=..., representation="v")``. The Cartesian
    field is ``sum_i v^i dX/de_i``, with the Jacobian of the mapping differentiated numerically
    from the attached ``X``, ``Y``, ``Z`` coordinates, so every logical direction needs at least
    two points.
    """
    data = _spatial(data, extra=("component",))
    if data.sizes["component"] != 3:
        raise ValueError(f"a vector field needs 3 components; got {data.sizes['component']}")
    short = [dim for dim in SPATIAL if data.sizes[dim] < 2]
    if short:
        raise ValueError(f"pushing forward needs at least two points along {short}")
    points = _points(data)
    axes = [np.asarray(data[dim], dtype=float) for dim in SPATIAL]
    jacobian = np.array(  # jacobian[a, i] = dX_a / de_i
        [[np.gradient(points[a], axes[i], axis=i) for i in range(3)] for a in range(3)]
    )
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


def _clim(values, clim):
    if clim is not None:
        return tuple(clim)
    finite = np.asarray(values)[np.isfinite(values)]
    if not finite.size:
        raise ValueError("cannot determine color limits without finite values; pass clim")
    return float(finite.min()), float(finite.max())


def _face(points, axis, index):
    face = [slice(None)] * 3
    face[axis] = slice(index, index + 1) if index >= 0 else slice(index, None)
    return points[tuple(face)]


def boundary_faces(points: np.ndarray) -> list[np.ndarray]:
    """The logical boundary faces of a ``(n1, n2, n3, 3)`` point array that are real boundaries.

    Faces that collapse to a line or point (a polar axis) and pairs of opposite faces that
    coincide (the seam of a periodic direction, e.g. ``phi = 0`` of a full torus) are dropped.
    A grid that is flat in one direction (a 2-D run) is its own single face.
    """
    scale = max(float(np.ptp(points.reshape(-1, 3), axis=0).max()), 1e-300)
    flat = [axis for axis in range(3) if points.shape[axis] == 1]
    if flat:
        return [points]
    faces = []
    for axis in range(3):
        low, high = _face(points, axis, 0), _face(points, axis, -1)
        if np.allclose(low, high, atol=1e-9 * scale):
            continue
        for face in (low, high):
            spread = np.ptp(face.reshape(-1, 3), axis=0)
            if np.sort(spread)[1] > 1e-9 * scale:  # spans a surface, not a line or point
                faces.append(face)
    return faces


def _grid_points(grid) -> np.ndarray:
    return np.asarray(grid.points).reshape((*grid.dimensions, 3), order="F")


def _add_context(plotter, grid, show_domain):
    if not show_domain:
        return
    pv = _pv()
    for i, face in enumerate(boundary_faces(_grid_points(grid))):
        surface = pv.StructuredGrid(face[..., 0], face[..., 1], face[..., 2])
        plotter.add_mesh(surface, color="lightgrey", opacity=0.12, name=f"domain{i}")


def _finish(plotter, title):
    if title:
        plotter.add_text(title, font_size=10, name="title")
    plotter.show_axes()
    return plotter


def pyvista_isosurface(
    data: xr.DataArray,
    *,
    values: int | list[float] = 5,
    cmap="viridis",
    opacity: float = 1.0,
    clim=None,
    show_domain: bool = True,
    title: str | None = None,
    plotter=None,
):
    """Contour surfaces of a selected scalar ``(e1, e2, e3)`` field in physical space.

    ``values`` is the number of evenly spaced levels, or explicit levels. ``show_domain`` draws
    the domain's outer surface translucently for context.
    """
    grid = structured_grid(data)
    name = grid.active_scalars_name
    lo, hi = _clim(grid[name], clim)
    levels = np.linspace(lo, hi, values + 2)[1:-1] if isinstance(values, int) else values
    plotter = _plotter(plotter)
    _add_context(plotter, grid, show_domain)
    contours = grid.contour(isosurfaces=list(levels), scalars=name)
    if contours.n_points:
        plotter.add_mesh(
            contours,
            scalars=name,
            cmap=cmap,
            clim=(lo, hi),
            opacity=opacity,
            scalar_bar_args={"title": value_label(data)},
            name="isosurface",
        )
    return _finish(plotter, _label(data) if title is None else title)


def _cut_indices(data, cuts):
    indices = {}
    for dim, positions in (cuts or {}).items():
        if dim not in SPATIAL:
            raise ValueError(f"cuts are along e1, e2 or e3; got {dim!r}")
        coordinate = np.asarray(data[dim], dtype=float)
        chosen = []
        for position in np.atleast_1d(positions).tolist():
            if position == "first":
                chosen.append(0)
            elif position == "last":
                chosen.append(len(coordinate) - 1)
            elif isinstance(position, int):
                chosen.append(position % len(coordinate))
            else:
                chosen.append(int(np.abs(coordinate - float(position)).argmin()))
        indices[dim] = chosen
    return indices


def prepare_slices_3d(data: xr.DataArray, *, cuts: dict | None = None) -> list[xr.DataArray]:
    """The logical cuts of a scalar ``(e1, e2, e3)`` field that :func:`pyvista_slices` draws.

    ``cuts`` maps ``e1``/``e2``/``e3`` to one position or a list: a float is the nearest
    logical coordinate, an integer a grid index, ``"first"``/``"last"`` an end. The default is
    the middle of every dimension with more than one point. Each cut keeps its size-one
    dimension, so it still maps onto a surface in physical space.
    """
    data = _spatial(data)
    if cuts is None:
        cuts = {dim: data.sizes[dim] // 2 for dim in SPATIAL if data.sizes[dim] > 1}
    return [
        data.isel({dim: [index]})
        for dim, indices in _cut_indices(data, cuts).items()
        for index in indices
    ]


def pyvista_slices(
    data: xr.DataArray,
    *,
    cuts: dict | None = None,
    cmap="viridis",
    clim=None,
    show_domain: bool = True,
    title: str | None = None,
    plotter=None,
):
    """Surfaces of constant logical coordinate through a scalar field, drawn in physical space.

    On a mapped domain these are the natural cuts: ``cuts={"e3": [0, 0.25]}`` gives poloidal
    cross-sections of a torus, ``cuts={"e1": 0.8}`` the field on one flux surface. See
    :func:`prepare_slices_3d` for ``cuts``; color limits are shared by every cut.
    """
    data = _spatial(data)
    pieces = prepare_slices_3d(data, cuts=cuts)
    lo, hi = _clim(np.asarray(data), clim)
    plotter = _plotter(plotter)
    _add_context(plotter, structured_grid(data), show_domain)
    for i, piece in enumerate(pieces):
        plotter.add_mesh(
            structured_grid(piece, name=_label(data)),
            cmap=cmap,
            clim=(lo, hi),
            scalar_bar_args={"title": value_label(data)},
            name=f"slice{i}",
        )
    return _finish(plotter, _label(data) if title is None else title)


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
    """Arrows of a selected ``(component, e1, e2, e3)`` vector field, colored by magnitude.

    ``components="cartesian"`` (default) for x/y/z components, e.g. a ``*_phy`` product;
    ``"contravariant"`` for logical components, pushed forward by :func:`push_forward`.
    ``stride`` thins the grid in every direction; ``scale`` is the arrow length of the largest
    vector (default: a tenth of the domain size).
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    name = _label(data)
    full = _vector_grid(data, components, name)
    thinned = _vector_grid(
        data.isel({dim: slice(None, None, stride) for dim in SPATIAL}), components, name
    )
    magnitude = thinned[f"|{name}|"]
    peak = float(magnitude.max()) if magnitude.size else 0.0
    length = 0.1 * full.length if scale is None else scale
    plotter = _plotter(plotter)
    _add_context(plotter, full, show_domain)
    if peak > 0:
        arrows = thinned.glyph(orient=name, scale=f"|{name}|", factor=length / peak)
        plotter.add_mesh(
            arrows, scalars=f"|{name}|", cmap=cmap, scalar_bar_args={"title": f"|{name}|"}, name="glyphs"
        )
    return _finish(plotter, name if title is None else title)


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

    Lines are traced in both directions from ``n_points`` seeds in a sphere of ``source_radius``
    (default: a quarter of the domain size) around ``source_center`` (default: the domain
    center). See :func:`pyvista_glyphs` for ``components``.
    """
    name = _label(data)
    grid = _vector_grid(data, components, name)
    lines = grid.streamlines(
        vectors=name,
        n_points=n_points,
        source_radius=0.25 * grid.length if source_radius is None else source_radius,
        source_center=grid.center if source_center is None else source_center,
        max_length=4 * grid.length if max_length is None else max_length,
        integration_direction="both",
    )
    plotter = _plotter(plotter)
    _add_context(plotter, grid, show_domain)
    if lines.n_points:
        if tube_radius:
            lines = lines.tube(radius=tube_radius)
        plotter.add_mesh(
            lines, scalars=f"|{name}|", cmap=cmap, line_width=2,
            scalar_bar_args={"title": f"|{name}|"}, name="streamlines",
        )
    return _finish(plotter, f"{name} field lines" if title is None else title)


def orbit_polylines(orbits: xr.Dataset, *, color_by: str = "t", max_markers: int = 200):
    """Marker orbits as one ``pyvista.PolyData`` line per marker, with point data ``color_by``.

    Samples where a marker is lost (every quantity zero) are dropped. ``color_by`` is ``"t"``,
    ``"classification"`` (see :func:`~struphy_plots.analysis.classify_orbits`), or the name of
    any ``(t, marker)`` variable, e.g. ``"v_par"`` or ``"weight"``.
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
        raise ValueError(
            f'color_by must be "t", "classification" or a variable of {tuple(subset.data_vars)}'
        )
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
            mesh, scalars=color_by, cmap=cmap or "viridis", line_width=2,
            scalar_bar_args={"title": color_by}, name="orbits",
        )
    return _finish(plotter, "Marker orbits" if title is None else title)


def pyvista_domain(
    domain,
    *,
    n1: int = 8,
    n2: int = 32,
    n3: int = 32,
    resolution: int = 4,
    color="black",
    surface: bool = True,
    title: str | None = None,
    plotter=None,
):
    """The mapping of a Struphy ``domain`` as a wireframe of logical grid lines.

    Grid lines are drawn on the logical boundary faces only (for a torus: the outer surface
    and the poloidal cross-section at ``e3 = 0``), ``n1``, ``n2``, ``n3`` per direction, each
    sampled ``resolution`` times finer so curved lines stay smooth. ``surface`` adds the
    translucent boundary. Useful to check the geometry (and its orientation) of a run.
    """
    pv = _pv()
    fine = [np.linspace(0.0, 1.0, (n - 1) * resolution + 1) for n in (n1, n2, n3)]
    x, y, z = (np.asarray(c, dtype=float) for c in domain(*fine, squeeze_out=False))
    grid = pv.StructuredGrid(x, y, z)
    plotter = _plotter(plotter)
    if surface:
        _add_context(plotter, grid, True)
    points = np.stack([x, y, z], axis=-1)
    segments, cells = [], []
    for axis in range(3):
        if points.shape[axis] < 2:
            continue
        others = [a for a in range(3) if a != axis]
        last = [points.shape[a] - 1 for a in others]
        for i in range(0, points.shape[others[0]], resolution):
            for j in range(0, points.shape[others[1]], resolution):
                if i not in (0, last[0]) and j not in (0, last[1]):
                    continue  # an interior line
                index = [slice(None)] * 3
                index[others[0]], index[others[1]] = i, j
                line = points[tuple(index)]
                if np.ptp(line, axis=0).max() == 0:  # a collapsed line, e.g. a polar axis
                    continue
                start = sum(len(s) for s in segments)
                segments.append(line)
                cells.append(np.concatenate([[len(line)], start + np.arange(len(line))]))
    if segments:
        wires = pv.PolyData(np.concatenate(segments), lines=np.concatenate(cells))
        plotter.add_mesh(wires, color=color, line_width=1, name="wireframe")
    return _finish(plotter, "Domain" if title is None else title)


RENDERERS = {
    "isosurface": pyvista_isosurface,
    "slices": pyvista_slices,
    "glyphs": pyvista_glyphs,
    "streamlines": pyvista_streamlines,
}


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
    """
    pv = _pv()
    if kind not in RENDERERS:
        raise ValueError(f"kind must be one of {tuple(RENDERERS)}; got {kind!r}")
    if not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError("step must be a positive integer")
    validate_array(data, required_dims=(sweep,))
    render = RENDERERS[kind]
    if kind in ("isosurface", "slices"):
        options["clim"] = _clim(np.asarray(data), clim)
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
                plotter.camera_position = "iso"
                plotter.reset_camera()
            plotter.write_frame()
    finally:
        plotter.close()
    return str(path)
