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
    "e1": r"$\eta_1$",
    "e2": r"$\eta_2$",
    "e3": r"$\eta_3$",
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
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"expected xarray.DataArray, got {type(data).__name__}")
    missing = tuple(dim for dim in required_dims if dim not in data.dims)
    if missing:
        raise ValueError(f"missing dimensions {missing}; available dimensions are {data.dims}")
    return data


def axis_label(data: xr.DataArray, dim: str) -> str:
    if dim not in data.dims:
        raise KeyError(f"dimension {dim!r} not found in {data.dims}")
    coord = data.coords.get(dim)
    label = ("" if coord is None else coord.attrs.get("long_name", "")) or DIM_LABELS.get(dim, dim)
    unit = "" if coord is None else coord.attrs.get("units", "")
    return f"{label} [{unit}]" if unit else label


def value_label(data: xr.DataArray) -> str:
    label = data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""
    unit = data.attrs.get("units", "") or "a.u."
    return f"{label} [{unit}]" if label else f"[{unit}]"


def scalar_names(scalars: xr.Dataset | Mapping, *, names=None, exclude=SCALARS_EXCLUDE) -> list[str]:
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
    """How a direction of a ``(..., 3)`` array of physical points wraps around, if it does:
    ``"closed"`` (the last slice repeats the first), ``"open"`` (it stops one step short of the
    seam, as on Struphy's cell centers), or ``None`` (not periodic, e.g. a radius or the ends of a
    torus sector)."""
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
    """``data`` with the first slice repeated at the end of every direction in ``dims`` that
    wraps around in physical space (default: those of ``e1``, ``e2``, ``e3`` it has).

    Struphy evaluates fields at cell centers, which leave out the seam of a periodic direction
    (e.g. the poloidal angle), so a surface or pcolormesh drawn through the points has a gap
    there. A direction counts as periodic when its last slice lies about one grid step from its
    first one; a radius, or the ends of a torus sector, do not. Needs the physical ``X``, ``Y``,
    ``Z`` coordinates; without them ``data`` comes back unchanged.
    """
    dims = [d for d in (dims or ("e1", "e2", "e3")) if d in data.dims]
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
    """d values / d coordinate along ``axis``.

    Around a periodic direction on a uniform grid (``kind`` ``"open"`` or ``"closed"``, see
    :func:`periodicity`) the derivative is spectral (an FFT), which is exact for the smooth,
    periodic angles of Struphy's mappings; elsewhere numpy's second-order differences (one-sided
    at the ends).
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
    """The Jacobian ``J[a, i] = dX_a / de_i`` of the mapping, shape ``(3, 3, n1, n2, n3)``.

    Differentiated numerically from the ``X``, ``Y``, ``Z`` coordinates of ``data`` on its
    ``(e1, e2, e3)`` grid: spectrally around periodic directions (see :func:`periodicity`),
    second order elsewhere, so every direction needs at least two points.
    """
    spatial = ("e1", "e2", "e3")
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
