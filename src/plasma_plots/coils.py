"""Stellarator coilsets and plasma boundaries as Datasets in plasma-plots' conventions.

The `CoilStellaration <https://huggingface.co/datasets/proxima-fusion/coilstellaration>`_
dataset stores a coilset by its unique coils: closed curves given by the Fourier coefficients of
their ``x``, ``y`` and ``z`` coordinates in the layout of DESC's ``FourierXYZCoil``, one current
per coil, the number of field periods and whether the coilset is stellarator symmetric. The plasma
boundaries come from `ConStellaration
<https://huggingface.co/datasets/proxima-fusion/constellaration>`_ as Fourier series of ``R`` and
``Z`` (``SurfaceRZFourier``).

* :func:`from_coilset` expands a coilset into every coil of the device, the way DESC's
  ``CoilSet`` does, and returns a Dataset over ``coil`` and the curve parameter ``s`` with
  ``X``, ``Y``, ``Z`` coordinates, the ``current`` of each coil, its ``length`` and its
  ``curvature`` along ``s``;
* :func:`from_boundary` evaluates a boundary into a Dataset over ``(rho, theta, zeta)`` with a
  single ``rho = 1``, in the layout of :func:`~plasma_plots.from_desc`, with the minor radius
  that CoilStellaration normalizes distances and curvatures by;
* :func:`boundary_from_desc` writes a DESC equilibrium's last surface in ConStellaration's
  layout;
* :func:`plot_boundary_sections` (``plasma.plasma.plot.boundary_sections()``) draws a
  boundary's cross-sections at several toroidal angles over half a field period;
* :func:`distance_to_surface` and :func:`coil_to_coil_distance` give, at every point of every
  coil, the distance to the plasma and to the nearest other coil: the benchmark's distance
  metrics are their minima;
* :func:`~plasma_plots.pyvista_plots.pyvista_coils` (``coils.plasma.plot.coils_3d()``) draws
  the coils as tubes around the plasma.

Neither function needs DESC or the ``coilstellaration`` package: plain dicts as stored in the
dataset's ``json`` columns are enough.

Examples
--------
>>> # a dict from the dataset's coilsets
>>> coils = plasma_plots.from_coilset(coilset)
>>> # ConStellaration's boundary.json
>>> plasma = plasma_plots.from_boundary(boundary)
>>> a = plasma.attrs["minor_radius"]
>>> float(coils.curvature.max()) * a  # normalized_coil_curvatures/max
>>> float(plasma_plots.coils.distance_to_surface(coils, plasma).min()) / a
>>> coils.plasma.plot.coils_3d(color_by="curvature", surface=plasma).show()
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import xarray as xr

from .mpi import rank_zero

#: The keys of a coilset in CoilStellaration's layout.
COILSET_KEYS = (
    "coil_x_n",
    "coil_y_n",
    "coil_z_n",
    "currents",
    "n_field_periods",
    "is_stellarator_symmetric",
)


def _as_mapping(value, keys, what):
    """A JSON string, a mapping or an object with attributes, as a dict of the given keys."""
    if isinstance(value, (str, bytes)):
        value = json.loads(value)
    if isinstance(value, Mapping):
        missing = [key for key in keys if key not in value]
        if missing:
            raise ValueError(f"{what} lacks {missing}")
        return {key: value[key] for key in keys}
    missing = [key for key in keys if not hasattr(value, key)]
    if missing:
        raise TypeError(
            f"expected {what} as a dict, a JSON string or an object with {list(keys)}; "
            f"{type(value).__name__} lacks {missing}"
        )
    return {key: getattr(value, key) for key in keys}


def _from_desc_coilset(coilset) -> dict:
    """A DESC ``CoilSet`` of ``FourierXYZCoil`` as a dict in CoilStellaration's layout."""
    coils = list(coilset.coils)
    if not all(hasattr(coil, "X_n") for coil in coils):
        kinds = sorted({type(coil).__name__ for coil in coils})
        raise TypeError(
            f"only coilsets of FourierXYZCoil are supported, got {kinds}; "
            "convert with coilset.to_FourierXYZ()"
        )
    orders = {len(np.asarray(coil.X_n)) for coil in coils}
    if len(orders) > 1:
        raise ValueError(f"the coils have different Fourier orders: {sorted(orders)}")
    return {
        "coil_x_n": [np.asarray(coil.X_n) for coil in coils],
        "coil_y_n": [np.asarray(coil.Y_n) for coil in coils],
        "coil_z_n": [np.asarray(coil.Z_n) for coil in coils],
        "currents": [float(np.squeeze(coil.current)) for coil in coils],
        "n_field_periods": int(coilset.NFP),
        "is_stellarator_symmetric": bool(coilset.sym),
    }


def _reflection(normal) -> np.ndarray:
    normal = np.asarray(normal, dtype=float)
    return np.eye(3) - 2.0 * np.outer(normal, normal)


def _fourier_basis(order: int, s: np.ndarray, derivative: int = 0) -> np.ndarray:
    """DESC's ``FourierXYZCoil`` basis ``[sin(N s), ..., sin(s), 1, cos(s), ..., cos(N s)]``.

    Returns the ``derivative``-th derivative with respect to ``s``, shape ``(2N + 1, len(s))``.
    """
    n = np.arange(-order, order + 1)[:, None]
    k = np.abs(n)
    phase = k * s[None, :] + derivative * np.pi / 2  # d/ds shifts sin and cos by pi/2
    sine = np.sin(phase)
    cosine = np.cos(phase)
    return np.where(n < 0, sine, cosine) * k.astype(float) ** derivative


def from_coilset(coilset, *, n_points: int = 128) -> xr.Dataset:
    """Every coil of a coilset as points along its curve, with currents and curvatures.

    The unique coils are expanded as DESC's ``CoilSet`` expands them: for a stellarator-symmetric
    coilset, their mirror images across the plane ``phi = pi / nfp`` (followed by ``z -> -z``) in
    reverse order, then all of these rotated into every field period. Coil ``i`` of the Dataset is
    coil ``i`` of DESC's expanded coilset.

    Parameters
    ----------
    coilset : dict, str, object or desc.coils.CoilSet
        A coilset in CoilStellaration's layout: ``coil_x_n``, ``coil_y_n``, ``coil_z_n``
        (``(n_unique_coils, 2N + 1)`` Fourier coefficients in meters, ordered
        ``[sin(N s), ..., sin(s), 1, cos(s), ..., cos(N s)]``), ``currents`` (A),
        ``n_field_periods`` and ``is_stellarator_symmetric``; as a dict (e.g. ``json.loads`` of
        the dataset's ``json`` column), that JSON string, an object with these attributes such as
        ``coilstellaration.types.Coilset``, or a DESC ``CoilSet`` of ``FourierXYZCoil``.
    n_points : int, optional
        Points along each coil, equally spaced in ``s`` over ``[0, 2π)``. Default: 128.

    Returns
    -------
    xarray.Dataset
        Over ``coil`` and ``s``: the coordinates ``X``, ``Y``, ``Z`` (m), ``unique_coil`` (the
        stored coil each one copies), ``mirrored`` and ``field_period``; the variables
        ``current`` (A, per coil), ``length`` (m, per coil) and ``curvature`` (1/m, along each
        coil); the attributes ``nfp``, ``stellarator_symmetric`` and ``fourier_order``.

    Raises
    ------
    TypeError
        If ``coilset`` is none of the accepted kinds.
    ValueError
        If keys are missing, the coefficient arrays don't match, or ``n_points < 3``.

    Examples
    --------
    >>> coils = from_coilset(json.loads(row["json"]))
    >>> coils.sizes  # 5 unique coils, mirrored, 3 field periods
    Frozen({'coil': 30, 's': 128})
    >>> coils.curvature.max(dim="s")  # the sharpest bend of each coil
    """
    if n_points < 3:
        raise ValueError(f"need at least 3 points per coil, got {n_points}")
    if hasattr(coilset, "coils") and hasattr(coilset, "NFP"):
        data = _from_desc_coilset(coilset)
    else:
        data = _as_mapping(coilset, COILSET_KEYS, "the coilset")
    arrays = [np.asarray(data[key], dtype=float) for key in COILSET_KEYS[:3]]
    shapes = {array.shape for array in arrays}
    if len(shapes) > 1 or arrays[0].ndim != 2 or arrays[0].shape[-1] % 2 != 1:
        raise ValueError(
            "coil_x_n, coil_y_n and coil_z_n must be (n_coils, 2N + 1) arrays of one shape, "
            f"got {sorted(shapes)}"
        )
    xyz_n = np.stack(arrays)  # (3, unique coil, mode)
    currents = np.atleast_1d(np.asarray(data["currents"], dtype=float))
    n_unique = xyz_n.shape[1]
    if currents.shape != (n_unique,):
        raise ValueError(
            f"need one current per coil ({n_unique}), got {currents.shape}"
        )
    nfp = int(data["n_field_periods"])
    sym = bool(data["is_stellarator_symmetric"])
    order = (xyz_n.shape[-1] - 1) // 2

    s = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    position, first, second = (
        np.einsum("dcm,ms->csd", xyz_n, _fourier_basis(order, s, k)) for k in range(3)
    )  # [unique coil, point, xyz]
    speed = np.linalg.norm(first, axis=-1)
    curvature = np.linalg.norm(np.cross(first, second), axis=-1) / speed**3
    length = speed.mean(axis=-1) * 2 * np.pi

    unique = np.arange(n_unique)
    mirrored = np.zeros(n_unique, dtype=bool)
    if sym:
        normal = [-np.sin(np.pi / nfp), np.cos(np.pi / nfp), 0.0]
        mirror = _reflection([0.0, 0.0, 1.0]) @ _reflection(normal)
        position = np.concatenate([position, (position @ mirror.T)[::-1]])
        curvature = np.concatenate([curvature, curvature[::-1]])
        length = np.concatenate([length, length[::-1]])
        currents = np.concatenate([currents, currents[::-1]])
        unique = np.concatenate([unique, unique[::-1]])
        mirrored = np.concatenate([mirrored, ~mirrored])
    per_period = position.shape[0]
    angles = 2 * np.pi * np.arange(nfp) / nfp
    rotations = np.array(
        [
            [[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]
            for a in angles
        ]
    )
    position = np.concatenate([position @ rotation.T for rotation in rotations])

    def tile(values):
        return np.tile(values, (nfp,) + (1,) * (np.ndim(values) - 1))

    dims = ("coil", "s")
    coords = {
        "coil": ("coil", np.arange(per_period * nfp)),
        "s": (
            "s",
            s,
            {"label": "$s$", "long_name": "curve parameter", "period": 2 * np.pi},
        ),
        **{
            axis: (dims, position[..., i], {"label": f"${axis}$", "units": "m"})
            for i, axis in enumerate("XYZ")
        },
        "unique_coil": ("coil", tile(unique), {"long_name": "stored coil it copies"}),
        "mirrored": (
            "coil",
            tile(mirrored),
            {"long_name": "stellarator-symmetric image"},
        ),
        "field_period": ("coil", np.repeat(np.arange(nfp), per_period)),
    }
    variables = {
        "current": (
            "coil",
            tile(currents),
            {"label": "$I$", "units": "A", "long_name": "coil current"},
        ),
        "length": (
            "coil",
            tile(length),
            {"label": "$L$", "units": "m", "long_name": "coil length"},
        ),
        "curvature": (
            dims,
            tile(curvature),
            {"label": r"$\kappa$", "units": "1/m", "long_name": "curvature"},
        ),
    }
    attrs = {"nfp": nfp, "stellarator_symmetric": int(sym), "fourier_order": order}
    return xr.Dataset(variables, coords=coords, attrs=attrs)


def _angles(value, stop: float, what: str) -> np.ndarray:
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if value < 1:
            raise ValueError(f"need at least one {what}, got {value}")
        return np.linspace(0.0, stop, int(value), endpoint=False)
    return np.atleast_1d(np.asarray(value, dtype=float))


def _surface_rz(surface: dict, theta: np.ndarray, phi: np.ndarray, derivative=False):
    """R and Z (or dR/dθ and dZ/dθ) of a ``SurfaceRZFourier`` on a (theta, phi) grid."""
    r_cos = np.asarray(surface["r_cos"], dtype=float)
    z_sin = np.asarray(surface["z_sin"], dtype=float)
    n_tor = (r_cos.shape[1] - 1) // 2
    m = np.arange(r_cos.shape[0])[:, None]
    n = np.arange(-n_tor, n_tor + 1)[None, :]
    nfp = int(surface["n_field_periods"])
    angle = (
        m[None, None] * theta[:, None, None, None]
        - nfp * n[None, None] * phi[None, :, None, None]
    )  # (theta, phi, m, n)
    cos, sin = np.cos(angle), np.sin(angle)
    if derivative:
        cos, sin = -m * sin, m * cos  # d/dθ
    r = np.sum(r_cos * cos, axis=(-2, -1))
    z = np.sum(z_sin * sin, axis=(-2, -1))
    if surface.get("r_sin") is not None:
        r = r + np.sum(np.asarray(surface["r_sin"], dtype=float) * sin, axis=(-2, -1))
    if surface.get("z_cos") is not None:
        z = z + np.sum(np.asarray(surface["z_cos"], dtype=float) * cos, axis=(-2, -1))
    return r, z


def from_boundary(boundary, *, theta=64, zeta=None) -> xr.Dataset:
    """A ConStellaration plasma boundary as a surface Dataset, with its minor radius.

    The boundary is a ``SurfaceRZFourier``: ``R`` and ``Z`` as Fourier series in the poloidal
    angle ``theta`` and the cylindrical toroidal angle ``phi``,
    ``R = sum r_cos[m, n] cos(m theta - nfp n phi)`` (plus ``r_sin``) and
    ``Z = sum z_sin[m, n] sin(m theta - nfp n phi)`` (plus ``z_cos``), with ``m = 0 ... M`` along
    the first axis and ``n = -N ... N`` along the second.

    The Dataset is laid out like one surface of :func:`~plasma_plots.from_desc`: dims ``rho``
    (the single value 1), ``theta`` and ``zeta`` (= ``phi``), coordinates ``X``, ``Y``, ``Z``,
    and the variable ``R``, so ``plasma.R.plasma.plot.slices_3d()`` draws it and it serves as the
    ``surface`` of :func:`~plasma_plots.pyvista_plots.pyvista_coils` and
    :func:`distance_to_surface`.

    Parameters
    ----------
    boundary : dict, str or object
        ``r_cos``, ``z_sin`` (``(M + 1, 2N + 1)``), optional ``r_sin``, ``z_cos`` (``None`` for a
        stellarator-symmetric surface) and ``n_field_periods``: ConStellaration's
        ``boundary.json`` as a dict or JSON string, or an object with these attributes such as
        ``constellaration.geometry.surface_rz_fourier.SurfaceRZFourier``.
    theta : int or sequence of float, optional
        Poloidal angles: an integer as that many points over ``[0, 2π)``, else the values.
        Default: 64.
    zeta : int or sequence of float, optional
        Toroidal angles: an integer as that many points over one field period ``[0, 2π/nfp)``,
        else the values. Default: the whole torus, 24 points per field period.

    Returns
    -------
    xarray.Dataset
        ``R`` (m) over ``(rho, theta, zeta)`` with coordinates ``X``, ``Y``, ``Z``, the angles'
        ``period``, and the attributes ``nfp``, ``major_radius`` (the ``m = n = 0``
        coefficient of ``R``, m) and ``minor_radius`` (m): ``sqrt(A / π)`` of the mean area
        ``A`` of the cross-sections at constant ``phi``, the length CoilStellaration's
        ``normalized_*`` metrics divide by.

    Raises
    ------
    TypeError
        If ``boundary`` is none of the accepted kinds.
    ValueError
        If keys are missing or ``r_cos`` has an even number of toroidal modes.

    Examples
    --------
    >>> plasma = from_boundary(json.loads(row["boundary.json"]))
    >>> plasma.attrs["minor_radius"]
    >>> plasma.R.plasma.plot.slices_3d().show()
    """
    if isinstance(boundary, (str, bytes)):
        boundary = json.loads(boundary)
    surface = _as_mapping(
        boundary, ("r_cos", "z_sin", "n_field_periods"), "the boundary"
    )
    for key in ("r_sin", "z_cos"):
        surface[key] = (
            boundary.get(key)
            if isinstance(boundary, Mapping)
            else getattr(boundary, key, None)
        )
    r_cos = np.asarray(surface["r_cos"], dtype=float)
    if r_cos.ndim != 2 or r_cos.shape[1] % 2 != 1:
        raise ValueError(
            f"r_cos must be an (M + 1, 2N + 1) array, got shape {r_cos.shape}"
        )
    nfp = int(surface["n_field_periods"])
    theta = _angles(theta, 2 * np.pi, "poloidal angle")
    zeta = (
        np.linspace(0.0, 2 * np.pi, 24 * nfp, endpoint=False)
        if zeta is None
        else _angles(zeta, 2 * np.pi / nfp, "toroidal angle")
    )
    R, Z = _surface_rz(surface, theta, zeta)

    # the mean cross-section area over one field period, by Green's theorem in each plane
    quad_theta = np.linspace(0.0, 2 * np.pi, 128, endpoint=False)
    quad_phi = np.linspace(0.0, 2 * np.pi / nfp, 33, endpoint=False)
    r_q, _ = _surface_rz(surface, quad_theta, quad_phi)
    _, dz_q = _surface_rz(surface, quad_theta, quad_phi, derivative=True)
    area = np.abs(np.mean(r_q * dz_q, axis=0) * 2 * np.pi).mean()
    n_tor = (r_cos.shape[1] - 1) // 2

    dims = ("rho", "theta", "zeta")

    def surface_field(values):
        return values[None]

    coords = {
        "rho": ("rho", [1.0], {"label": r"$\rho$", "long_name": "plasma boundary"}),
        "theta": (
            "theta",
            theta,
            {"label": r"$\theta$", "long_name": "poloidal angle", "period": 2 * np.pi},
        ),
        "zeta": (
            "zeta",
            zeta,
            {
                "label": r"$\phi$",
                "long_name": "toroidal angle",
                "period": 2 * np.pi / nfp,
            },
        ),
        "X": (dims, surface_field(R * np.cos(zeta)), {"label": "$X$", "units": "m"}),
        "Y": (dims, surface_field(R * np.sin(zeta)), {"label": "$Y$", "units": "m"}),
        "Z": (dims, surface_field(Z), {"label": "$Z$", "units": "m"}),
    }
    variables = {
        "R": (
            dims,
            surface_field(R),
            {"label": "$R$", "units": "m", "long_name": "major radius", "nfp": nfp},
        )
    }
    attrs = {
        "nfp": nfp,
        "major_radius": float(r_cos[0, n_tor]),
        "minor_radius": float(np.sqrt(area / np.pi)),
    }
    return xr.Dataset(variables, coords=coords, attrs=attrs)


def boundary_from_desc(eq, *, max_poloidal_mode=None, max_toroidal_mode=None) -> dict:
    """The last flux surface of a DESC equilibrium in ConStellaration's boundary layout.

    ``R`` and ``Z`` of the surface ``rho = 1``, evaluated on a grid of DESC's angles (its
    ``zeta`` is the cylindrical angle ``phi``), are transformed into the coefficients of
    ``cos(m theta - nfp n phi)`` and ``sin(m theta - nfp n phi)``. With enough modes (the default:
    DESC's own resolution of the surface) the series reproduces the surface to round-off, so
    :func:`from_boundary` of the result is DESC's surface. Only stellarator-symmetric
    equilibria are supported.

    Parameters
    ----------
    eq : desc.equilibrium.Equilibrium
        The equilibrium, e.g. ``desc.examples.get("W7-X")``.
    max_poloidal_mode, max_toroidal_mode : int, optional
        ``M`` and ``N`` of the series. Default: DESC's ``eq.surface.M`` and ``eq.surface.N``.

    Returns
    -------
    dict
        ``r_cos`` and ``z_sin`` (nested lists, ``(M + 1, 2N + 1)``), ``r_sin`` and ``z_cos``
        (``None``), ``n_field_periods`` and ``is_stellarator_symmetric``: the layout of
        ConStellaration's ``boundary.json``.

    Raises
    ------
    ValueError
        If the equilibrium isn't stellarator symmetric.

    Examples
    --------
    >>> boundary = boundary_from_desc(desc.examples.get("W7-X"))
    >>> from_boundary(boundary).attrs["minor_radius"]  # DESC's "a"
    """
    from desc.grid import LinearGrid

    if not bool(eq.sym):
        raise ValueError("only stellarator-symmetric equilibria are supported")
    M = int(eq.surface.M if max_poloidal_mode is None else max_poloidal_mode)
    N = int(eq.surface.N if max_toroidal_mode is None else max_toroidal_mode)
    nfp = int(eq.NFP)
    n_theta, n_zeta = 4 * M + 4, 4 * N + 4
    grid = LinearGrid(rho=np.array([1.0]), theta=n_theta, zeta=n_zeta, NFP=nfp)
    values = eq.compute(["R", "Z"], grid=grid)
    # DESC's node order: zeta slowest, theta fastest
    R = np.asarray(values["R"]).reshape(n_zeta, n_theta).T
    Z = np.asarray(values["Z"]).reshape(n_zeta, n_theta).T
    # R = sum_{k, l} c_kl exp(i (k theta + l nfp phi)), and cos(m theta - n nfp phi) pairs
    # c_{m, -n} with its conjugate c_{-m, n}
    r_hat = np.fft.fft2(R) / R.size
    z_hat = np.fft.fft2(Z) / Z.size
    r_cos = np.zeros((M + 1, 2 * N + 1))
    z_sin = np.zeros((M + 1, 2 * N + 1))
    for m in range(M + 1):
        for n in range(-N, N + 1):
            if m == 0 and n < 0:
                continue  # the same modes as m = 0, n > 0
            k, l = m % n_theta, (-n) % n_zeta
            factor = 1.0 if m == 0 and n == 0 else 2.0
            r_cos[m, n + N] = factor * r_hat[k, l].real
            z_sin[m, n + N] = -factor * z_hat[k, l].imag
    return {
        "r_cos": r_cos.tolist(),
        "z_sin": z_sin.tolist(),
        "r_sin": None,
        "z_cos": None,
        "n_field_periods": nfp,
        "is_stellarator_symmetric": True,
    }


def _xyz(data) -> np.ndarray:
    """The ``X``, ``Y``, ``Z`` coordinates of ``data`` as an ``(n, 3)`` array."""
    missing = [axis for axis in "XYZ" if axis not in data.coords]
    if missing:
        raise ValueError(f"need the physical coordinates X, Y, Z; missing {missing}")
    arrays = xr.broadcast(*(data.coords[axis] for axis in "XYZ"))
    return np.stack([np.asarray(a, dtype=float).ravel() for a in arrays], axis=-1)


def _nearest(points: np.ndarray, targets: np.ndarray, chunk: int = 512) -> np.ndarray:
    """The distance from each of ``points`` to the nearest of ``targets``."""
    out = np.empty(len(points))
    for start in range(0, len(points), chunk):
        block = points[start : start + chunk]
        squared = (
            np.sum(block**2, axis=1)[:, None]
            - 2.0 * block @ targets.T
            + np.sum(targets**2, axis=1)[None, :]
        )
        out[start : start + chunk] = np.sqrt(np.maximum(squared.min(axis=1), 0.0))
    return out


def _coil_points(coils: xr.Dataset) -> np.ndarray:
    if not {"coil", "s"} <= set(coils.dims):
        raise ValueError(
            f"need a coilset Dataset over (coil, s), got dims {tuple(coils.dims)}"
        )
    return np.stack(
        [
            np.asarray(coils.coords[axis].transpose("coil", "s"), dtype=float)
            for axis in "XYZ"
        ],
        axis=-1,
    )


def distance_to_surface(coils: xr.Dataset, surface) -> xr.DataArray:
    """The distance from every point of every coil to the nearest point of a surface.

    The surface is sampled at its grid points, so the distance is an upper bound that converges
    as the surface grid is refined: with spacing ``h`` it is within about ``h² / (8 r)`` of the
    exact one, ``r`` the distance itself.

    Parameters
    ----------
    coils : xarray.Dataset
        Coils from :func:`from_coilset`.
    surface : xarray.Dataset or xarray.DataArray
        Anything with ``X``, ``Y``, ``Z`` coordinates: :func:`from_boundary`, or the last
        surface of :func:`~plasma_plots.from_desc` (``rho=[1.0]`` over the whole torus).

    Returns
    -------
    xarray.DataArray
        ``plasma_distance`` (m) over ``(coil, s)``; its minimum is the coil–plasma distance.

    Raises
    ------
    ValueError
        If ``coils`` isn't over ``(coil, s)`` or ``surface`` lacks ``X``, ``Y``, ``Z``.

    Examples
    --------
    >>> d = distance_to_surface(coils, plasma)
    >>> # normalized_coil_to_plasma_min_distances
    >>> float(d.min()) / plasma.attrs["minor_radius"]
    """
    points = _coil_points(coils)
    distance = _nearest(points.reshape(-1, 3), _xyz(surface)).reshape(points.shape[:2])
    return xr.DataArray(
        distance,
        dims=("coil", "s"),
        coords={
            name: coord
            for name, coord in coils.coords.items()
            if set(coord.dims) <= {"coil", "s"}
        },
        name="plasma_distance",
        attrs={"label": "$d_p$", "units": "m", "long_name": "distance to the plasma"},
    )


def coil_to_coil_distance(coils: xr.Dataset) -> xr.DataArray:
    """The distance from every point of every coil to the nearest point of any other coil.

    Coils are sampled at their points, so as for :func:`distance_to_surface` the distance
    converges from above as ``n_points`` of :func:`from_coilset` grows.

    Parameters
    ----------
    coils : xarray.Dataset
        Coils from :func:`from_coilset`, with at least two coils.

    Returns
    -------
    xarray.DataArray
        ``coil_distance`` (m) over ``(coil, s)``; its minimum is the coil–coil distance.

    Raises
    ------
    ValueError
        If ``coils`` isn't over ``(coil, s)`` or has a single coil.

    Examples
    --------
    >>> d = coil_to_coil_distance(coils)
    >>> # normalized_coil_to_coil_min_distances
    >>> float(d.min()) / plasma.attrs["minor_radius"]
    """
    points = _coil_points(coils)
    n_coils = points.shape[0]
    if n_coils < 2:
        raise ValueError("need at least two coils")
    distance = np.empty(points.shape[:2])
    for i in range(n_coils):
        others = np.concatenate([points[:i], points[i + 1 :]]).reshape(-1, 3)
        distance[i] = _nearest(points[i], others)
    return xr.DataArray(
        distance,
        dims=("coil", "s"),
        coords={
            name: coord
            for name, coord in coils.coords.items()
            if set(coord.dims) <= {"coil", "s"}
        },
        name="coil_distance",
        attrs={
            "label": "$d_c$",
            "units": "m",
            "long_name": "distance to the nearest other coil",
        },
    )


def _section_planes(surface: xr.Dataset, planes: int | None) -> xr.Dataset:
    """The surface at ``planes`` toroidal angles over half a field period (nearest grid points)."""
    if planes is None:
        return surface
    if planes < 1:
        raise ValueError(f"need at least one plane, got {planes}")
    nfp = int(surface.attrs.get("nfp", surface["R"].attrs.get("nfp", 1)))
    wanted = np.linspace(0.0, np.pi / nfp, planes)
    return surface.sel(zeta=wanted, method="nearest")


def _plane_label(phi: float, nfp: int) -> str:
    """``phi`` as a fraction of ``π / nfp``, e.g. ``$\\phi = 1/4\\ \\pi / N_{fp}$``."""
    from fractions import Fraction

    fraction = phi / (np.pi / nfp)
    exact = Fraction(fraction).limit_denominator(12)
    if abs(float(exact) - fraction) > 1e-6:
        text = f"{fraction:.2f}"
    elif exact == 0:
        return r"$\phi = 0$"
    elif exact == 1:
        text = ""
    else:
        text = f"{exact.numerator}/{exact.denominator}\\ "
    return rf"$\phi = {text}\pi / N_{{fp}}$"


@rank_zero
def plot_boundary_sections(
    boundary,
    *,
    planes: int | None = 5,
    theta: int = 128,
    ax=None,
    cmap="viridis",
    title: str | None = None,
):
    """Draw a plasma boundary's cross-sections in the ``(R, Z)`` plane at several toroidal angles.

    The view ConStellaration draws of its boundaries: by default five planes from ``phi = 0`` to
    half a field period, ``phi = π / nfp``, which for a stellarator-symmetric surface show every
    shape it takes.

    Parameters
    ----------
    boundary : dict, str, object or xarray.Dataset
        A boundary as :func:`from_boundary` takes it (ConStellaration's ``boundary.json``), or a
        surface Dataset with ``R`` and ``Z`` over ``theta`` and ``zeta`` and a single ``rho``,
        such as :func:`from_boundary`'s or the last surface of :func:`~plasma_plots.from_desc`.
    planes : int or None, optional
        The number of toroidal angles, evenly from 0 to ``π / nfp``; for a Dataset the nearest
        of its angles. ``None`` draws every angle of a Dataset. Default: 5.
    theta : int, optional
        Poloidal points per cross-section of a boundary (not of a Dataset). Default: 128.
    ax : matplotlib.axes.Axes, optional
        The axes to draw into. Default: a new figure.
    cmap : str or matplotlib colormap, optional
        The colors of the planes, from the first to the last. Default: ``"viridis"``.
    title : str, optional
        The axes title. Default: none.

    Returns
    -------
    PlotResult
        The figure, the axes, one closed line per plane, and in ``data["surface"]`` the
        surface at the drawn angles.

    Raises
    ------
    ValueError
        If ``planes`` is below 1, or the Dataset lacks ``R``, ``Z`` or the ``theta`` and
        ``zeta`` dimensions.

    Examples
    --------
    >>> plot_boundary_sections(json.loads(row["boundary.json"]))
    >>> plasma_plots.from_boundary(boundary).plasma.plot.boundary_sections(planes=3)
    """
    import matplotlib.pyplot as plt

    from .plotting import PlotResult, _finish, _plot_style

    if isinstance(boundary, xr.Dataset):
        missing = {"R", "Z"} - set(boundary.variables)
        missing |= {"theta", "zeta"} - set(boundary.dims)
        if missing:
            raise ValueError(
                f"the surface Dataset lacks {sorted(missing)}: it needs R and Z over theta and zeta"
            )
        surface = _section_planes(boundary, planes)
    else:
        if planes is not None and planes < 1:
            raise ValueError(f"need at least one plane, got {planes}")
        if isinstance(boundary, (str, bytes)):
            boundary = json.loads(boundary)
        nfp = int(
            boundary["n_field_periods"]
            if isinstance(boundary, Mapping)
            else boundary.n_field_periods
        )
        surface = from_boundary(
            boundary,
            theta=theta,
            zeta=np.linspace(0.0, np.pi / nfp, 5 if planes is None else planes),
        )
    if "rho" in surface.dims:
        surface = surface.isel(rho=0, drop=True)
    R = surface["R"].transpose("theta", "zeta")
    Z = surface["Z"].transpose("theta", "zeta")
    nfp = int(surface.attrs.get("nfp", surface["R"].attrs.get("nfp", 1)))
    zeta = np.atleast_1d(surface["zeta"].values)
    colors = plt.get_cmap(cmap)(np.linspace(0.0, 1.0, max(len(zeta), 2)))
    with _plot_style():
        own = ax is None
        if own:
            fig, ax = plt.subplots(figsize=(6.0, 5.0))
        else:
            fig = ax.figure
        lines = []
        for k, phi in enumerate(zeta):
            r, z = R.values[:, k], Z.values[:, k]
            (line,) = ax.plot(
                np.append(r, r[0]),
                np.append(z, z[0]),
                color=colors[k],
                label=_plane_label(phi, nfp),
            )
            lines.append(line)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set(xlabel="$R$ (m)", ylabel="$Z$ (m)")
        if title:
            ax.set_title(title)
        ax.legend(fontsize="small", loc="center left", bbox_to_anchor=(1.02, 0.5))
        _finish(fig, tight=own)
    return PlotResult(fig, ax, lines, data={"surface": surface})
