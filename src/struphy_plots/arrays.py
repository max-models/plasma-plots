"""Small xarray metadata helpers used by :mod:`struphy_plots`.

They intentionally live here rather than in Struphy so the plotting package can
operate on labeled xarray data from any producer.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr

DIM_LABELS = {
    "t": r"$t$",
    "eta1": r"$\eta_1$",
    "eta2": r"$\eta_2$",
    "eta3": r"$\eta_3$",
    "v1": r"$v_1$",
    "v2": r"$v_2$",
    "v3": r"$v_3$",
    "x": r"$x$",
    "y": r"$y$",
    "z": r"$z$",
    "R": r"$R$",
    "Z": r"$Z$",
    "component": "component",
    "marker": "marker",
    "quantity": "quantity",
}
SCALARS_EXCLUDE = ("time",)


def validate_array(data: xr.DataArray, *, required_dims: Sequence[str] = ()) -> xr.DataArray:
    """Check that ``data`` is an xarray.DataArray with the required dimensions.

    Parameters
    ----------
    data : xarray.DataArray
        The array to check.
    required_dims : sequence of str, optional
        Dimensions ``data`` must have. Default: none.

    Returns
    -------
    xarray.DataArray
        ``data`` itself, unchanged.

    Raises
    ------
    TypeError
        If ``data`` is not an xarray.DataArray.
    ValueError
        If one of ``required_dims`` is missing.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"expected xarray.DataArray, got {type(data).__name__}")
    missing = tuple(dim for dim in required_dims if dim not in data.dims)
    if missing:
        raise ValueError(f"missing dimensions {missing}; available dimensions are {data.dims}")
    return data


def axis_label(data: xr.DataArray, dim: str) -> str:
    """The axis label of a dimension, with its units.

    The label is the coordinate's ``long_name`` attribute, else a default for the known
    dimensions (matplotlib mathtext such as ``$\\eta_1$`` for ``eta1``), else the dimension name.

    Parameters
    ----------
    data : xarray.DataArray
        The array.
    dim : str
        One of its dimensions.

    Returns
    -------
    str
        The label, followed by ``[units]`` when the coordinate has a ``units`` attribute.

    Raises
    ------
    KeyError
        If ``dim`` is not a dimension of ``data``.
    """
    if dim not in data.dims:
        raise KeyError(f"dimension {dim!r} not found in {data.dims}")
    coord = data.coords.get(dim)
    label = ("" if coord is None else coord.attrs.get("long_name", "")) or DIM_LABELS.get(dim, dim)
    unit = "" if coord is None else coord.attrs.get("units", "")
    return f"{label} [{unit}]" if unit else label


def value_label(data: xr.DataArray) -> str:
    """The label of an array's values, with their units.

    Parameters
    ----------
    data : xarray.DataArray
        The array. The label is its ``label`` attribute, else ``long_name``, else its name.

    Returns
    -------
    str
        ``"label [units]"``, with the ``units`` attribute or ``a.u.``; only ``"[units]"`` when
        there is no label.
    """
    label = data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""
    unit = data.attrs.get("units", "") or "a.u."
    return f"{label} [{unit}]" if label else f"[{unit}]"


def scalar_names(scalars: xr.Dataset | Mapping, *, names=None, exclude=SCALARS_EXCLUDE) -> list[str]:
    """The names of the scalar time series to use from a collection.

    Parameters
    ----------
    scalars : xarray.Dataset or mapping
        The scalars, e.g. ``out.scalars``.
    names : sequence of str, optional
        Explicit names, which must all be present; ``exclude`` then does not apply. Default:
        every variable not in ``exclude``.
    exclude : sequence of str, optional
        Names left out when ``names`` is not given. Default: ``("time",)``.

    Returns
    -------
    list of str
        The selected names, in order.

    Raises
    ------
    KeyError
        If one of ``names`` is not in ``scalars``.
    """
    available = tuple(scalars.data_vars if isinstance(scalars, xr.Dataset) else scalars.keys())
    if names is not None:
        missing = [name for name in names if name not in available]
        if missing:
            raise KeyError(f"no scalars {missing}, available: {available}")
        return list(names)
    return [name for name in available if name not in exclude]


def save_scalars(
    scalars: xr.Dataset | Mapping,
    path: str,
    *,
    names=None,
    exclude=SCALARS_EXCLUDE,
    fmt=None,
) -> str:
    """Save scalar time series to a CSV or NPZ file.

    Parameters
    ----------
    scalars : xarray.Dataset or mapping
        The scalars, e.g. ``out.scalars``. Each selected one must have ``t`` as its only
        dimension, with identical time coordinates.
    path : str
        The file to write.
    names : sequence of str, optional
        The scalars to save. Default: every one not in ``exclude``; see :func:`scalar_names`.
    exclude : sequence of str, optional
        Names left out when ``names`` is not given. Default: ``("time",)``.
    fmt : {"csv", "npz"}, optional
        The file format. Default: from the extension of ``path``, else ``"csv"``. A CSV has a
        header line ``t,<names>`` and one row per time; an NPZ has the arrays ``t`` and one per
        name.

    Returns
    -------
    str
        ``path``.

    Raises
    ------
    ValueError
        If a scalar has dimensions other than ``t``, the time coordinates differ, or the format
        is unknown.
    KeyError
        If one of ``names`` is not in ``scalars``.

    Examples
    --------
    >>> save_scalars(out.scalars, "scalars.csv", names=["en_E", "en_tot"])
    """
    selected = scalar_names(scalars, names=names, exclude=exclude)
    arrays = [scalars[name] for name in selected]
    for array in arrays:
        validate_array(array, required_dims=("t",))
        if array.dims != ("t",):
            raise ValueError(f"scalar {array.name!r} must have only the 't' dimension, got {array.dims}")
    if arrays:
        arrays = xr.align(*arrays, join="exact")
        time = np.asarray(arrays[0].coords["t"])
        values = np.column_stack([np.asarray(array) for array in arrays])
    else:
        time, values = np.zeros(0), np.zeros((0, 0))
    fmt = (fmt or os.path.splitext(path)[1].lstrip(".") or "csv").lower()
    if fmt == "npz":
        np.savez(path, t=time, **{name: values[:, i] for i, name in enumerate(selected)})
    elif fmt == "csv":
        np.savetxt(
            path,
            np.column_stack((time, values)),
            delimiter=",",
            header=",".join(("t", *selected)),
            comments="",
        )
    else:
        raise ValueError(f"unknown format {fmt!r}, expected 'csv' or 'npz'")
    return path


def periodicity(points: np.ndarray, axis: int) -> str | None:
    """How a direction of a ``(..., 3)`` array of physical points wraps around, if it does.

    The direction is ``"closed"`` when its last slice coincides with its first (to 1e-9 of the
    median step), and ``"open"`` when the seam between them is at most 1.5 times the last grid
    step everywhere.

    Parameters
    ----------
    points : numpy.ndarray
        Physical points, with the Cartesian coordinates ``(X, Y, Z)`` along the last axis.
    axis : int
        The axis of ``points`` to examine; it needs at least three points to be periodic.

    Returns
    -------
    {"closed", "open", None}
        ``"closed"`` (the last slice repeats the first), ``"open"`` (it stops one step short of
        the seam, as on Struphy's cell centers), or ``None`` (not periodic, e.g. a radius or the
        ends of a torus sector).
    """
    n = points.shape[axis]
    if n < 3:
        return None
    take = lambda i: np.take(points, i, axis=axis)  # noqa: E731
    seam = np.linalg.norm(take(-1) - take(0), axis=-1)
    step = np.linalg.norm(take(-1) - take(-2), axis=-1)
    scale = np.median(step)
    if scale <= 0:
        return None
    if np.all(seam < 1e-9 * scale):
        return "closed"
    if np.all(seam <= 1.5 * np.maximum(step, 1e-12 * scale)):
        return "open"
    return None


def close_periodic(data: xr.DataArray, dims=None) -> xr.DataArray:
    """Repeat the first slice at the end of every direction that wraps around in physical space.

    Struphy evaluates fields at cell centers, which leave out the seam of a periodic direction
    (e.g. the poloidal angle), so a surface or pcolormesh drawn through the points has a gap
    there. A direction counts as periodic when its last slice lies about one grid step from its
    first one (``"open"`` in :func:`periodicity`); a radius, or the ends of a torus sector, do
    not. The repeated slice gets the logical coordinate one step past the last.

    Parameters
    ----------
    data : xarray.DataArray
        The field, with the physical ``X``, ``Y``, ``Z`` coordinates; without them ``data``
        comes back unchanged.
    dims : sequence of str, optional
        The directions to close, if periodic. Default: those of ``eta1``, ``eta2``, ``eta3``
        that ``data`` has.

    Returns
    -------
    xarray.DataArray
        ``data``, one point longer along each periodic direction in ``dims``.
    """
    dims = [d for d in (dims or ("eta1", "eta2", "eta3")) if d in data.dims]
    if not dims or any(name not in data.coords for name in ("X", "Y", "Z")):
        return data

    def points(array):
        stacked = []
        for name in ("X", "Y", "Z"):
            coordinate = array.coords[name]
            for dim in dims:
                if dim not in coordinate.dims:
                    coordinate = coordinate.expand_dims({dim: array.sizes[dim]})
            stacked.append(np.asarray(coordinate.transpose(*dims), dtype=float))
        return np.stack(stacked, axis=-1)

    current = points(data)
    for axis, dim in enumerate(dims):
        if periodicity(current, axis) == "open":
            first = data.isel({dim: [0]})
            step = float(data[dim][1] - data[dim][0]) if dim in data.coords else 1.0
            end = float(data[dim][-1]) + step if dim in data.coords else float(data.sizes[dim])
            data = xr.concat(
                [data, first.assign_coords({dim: [end]})],
                dim=dim,
                coords="minimal",
                compat="override",
            )
            current = points(data)
    return data


def logical_derivative(values: np.ndarray, coordinate: np.ndarray, axis: int, kind: str | None) -> np.ndarray:
    """The derivative ``∂ values / ∂ coordinate`` along ``axis``.

    Around a periodic direction on a uniform grid (``kind`` ``"open"`` or ``"closed"``, see
    :func:`periodicity`) the derivative is spectral (an FFT), which is exact for the smooth,
    periodic angles of Struphy's mappings; the Nyquist mode of an even number of points is
    dropped. Elsewhere numpy's second-order differences (one-sided at the ends; first order with
    only two points).

    Parameters
    ----------
    values : numpy.ndarray
        The values to differentiate.
    coordinate : numpy.ndarray
        The coordinate along ``axis``, one value per point.
    axis : int
        The axis of ``values`` to differentiate along.
    kind : {"open", "closed", None}
        How the direction wraps around, from :func:`periodicity`: ``"open"`` (the seam is left
        out), ``"closed"`` (the last point repeats the first) or ``None`` (not periodic).

    Returns
    -------
    numpy.ndarray
        The derivative, of the same shape as ``values``.
    """
    spacing = np.diff(coordinate)
    uniform = len(spacing) > 0 and np.allclose(spacing, spacing[0])
    if kind is None or not uniform:
        return np.gradient(values, coordinate, axis=axis, edge_order=2 if len(coordinate) > 2 else 1)
    unique = np.take(values, range(values.shape[axis] - 1), axis=axis) if kind == "closed" else values
    n = unique.shape[axis]
    wavenumbers = 2 * np.pi * np.fft.fftfreq(n, d=spacing[0])
    if n % 2 == 0:
        wavenumbers[n // 2] = 0.0  # the Nyquist mode has no well-defined derivative
    shape = [1] * unique.ndim
    shape[axis] = n
    derivative = np.fft.ifft(1j * wavenumbers.reshape(shape) * np.fft.fft(unique, axis=axis), axis=axis).real
    if kind == "closed":
        derivative = np.concatenate([derivative, np.take(derivative, [0], axis=axis)], axis=axis)
    return derivative


def mapping_jacobian(data: xr.DataArray) -> np.ndarray:
    """The Jacobian ``J[a, i] = ∂X_a / ∂η_i`` of the mapping, shape ``(3, 3, n1, n2, n3)``.

    Differentiated numerically from the ``X``, ``Y``, ``Z`` coordinates of ``data`` on its
    ``(eta1, eta2, eta3)`` grid: spectrally around periodic directions (see :func:`periodicity`),
    second order elsewhere (see :func:`logical_derivative`).

    Parameters
    ----------
    data : xarray.DataArray
        An array with the dimensions ``eta1``, ``eta2``, ``eta3``, each with at least two
        points, and the ``X``, ``Y``, ``Z`` coordinates over them.

    Returns
    -------
    numpy.ndarray
        ``J`` of shape ``(3, 3, n1, n2, n3)``: the Cartesian component ``a`` first, the logical
        direction ``i`` second.

    Raises
    ------
    ValueError
        If a logical dimension is missing or has fewer than two points (pass a struphy domain to
        the calling function instead).
    """
    spatial = ("eta1", "eta2", "eta3")
    missing = [d for d in spatial if d not in data.dims]
    short = [d for d in spatial if d in data.dims and data.sizes[d] < 2]
    if missing or short:
        raise ValueError(
            f"a numerical Jacobian needs at least two points along each of {spatial}; "
            f"{missing + short} have fewer (pass a struphy domain instead)"
        )
    points = [np.asarray(data.coords[name].transpose(*spatial), dtype=float) for name in ("X", "Y", "Z")]
    axes = [np.asarray(data[d], dtype=float) for d in spatial]
    kinds = [periodicity(np.stack(points, axis=-1), axis) for axis in range(3)]
    return np.array([[logical_derivative(points[a], axes[i], i, kinds[i]) for i in range(3)] for a in range(3)])
