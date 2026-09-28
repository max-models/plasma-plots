"""Evaluate DESC equilibria into Datasets in plasma-plots' conventions.

`DESC <https://desc-docs.readthedocs.io>`_ computes a quantity of an equilibrium on a grid of
flux coordinates, ``eq.compute(names, grid)``, and returns flat arrays over the grid's nodes.
:func:`from_desc` evaluates an equilibrium on a tensor-product grid and returns an
``xarray.Dataset`` that the rest of plasma-plots reads the way it reads GVEC's:

* the dimensions are the flux coordinates ``rho``, ``theta`` (DESC's poloidal angle, or the PEST
  angle ``theta_P`` with ``sfl="pest"``) and ``zeta`` (the cylindrical toroidal angle), so
  ``rho=0.5`` selects and ``x="zeta"`` draws;
* ``X``, ``Y``, ``Z`` are coordinates of every variable (for ``coords="physical"``, 3-D views and
  the vector calculus on the mapped domain), and ``theta_P`` too when ``"theta_PEST"`` is among
  the names (for the ``coordinate_lines`` overlay);
* vectors, which DESC gives in cylindrical components, are Cartesian ``(x, y, z)`` along
  ``component``, as on every other grid;
* profiles (DESC's quantities over ``rho`` only) are over ``rho``, and global quantities such as
  ``"V"`` are scalars;
* each quantity keeps DESC's name (``ds["|B|"]``) and gets DESC's LaTeX ``label``, its ``units``
  and its description as ``long_name``;
* the angles get a ``period`` attribute (2π, and 2π/nfp toroidally), from which the mode spectra
  take their periods and full-torus mode numbers, and the ``coordinate_lines`` overlay its lines;
* the number of field periods is the ``nfp`` attribute of the Dataset and of every variable.

Integrals over the grid cover the sampled toroidal range, one field period by default, as for
GVEC: multiply by nfp for the device. DESC's Jacobian is ``"sqrt(g)"``. Unlike GVEC's, the
evaluation needs DESC itself (``pip install desc-opt``); the Dataset it returns does not
(``to_netcdf`` saves it).

Examples
--------
>>> import desc.examples
>>> import plasma_plots
>>> eq = desc.examples.get("W7-X")
>>> ev = plasma_plots.from_desc(eq, ["|B|", "iota", "sqrt(g)"], rho=11, theta=64, zeta=40)
>>> ev["|B|"].plasma.plot.slice(coords="physical", plane="RZ", zeta=0.0,
...                             overlays={"coordinate_lines": {"rho": 5, "theta": 8}})
>>> ev.iota.plasma.plot.lineout(rationals=4)
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import xarray as xr

#: The quantities every evaluation computes, for the coordinates and the vector components.
GEOMETRY = ("X", "Y", "Z", "phi", "rho", "zeta")
#: The DESC quantities that become coordinates rather than variables, and their names here.
GRID_QUANTITIES = {"theta_PEST": "theta_P"}
POLOIDAL = {None: "theta", "pest": "theta_P"}
CARTESIAN = ["x", "y", "z"]


def _points(value, stop: float, *, endpoint: bool) -> np.ndarray:
    """An integer as that many equally spaced points in ``[0, stop]``, anything else as values."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if value < 1:
            raise ValueError(f"need at least one point, got {value}")
        return np.linspace(0.0, stop, int(value), endpoint=endpoint)
    return np.atleast_1d(np.asarray(value, dtype=float))


def _label(latex: str) -> str:
    """A DESC label as a mathtext label that Matplotlib and VTK both render."""
    for sizing in (r"\left", r"\right"):
        latex = latex.replace(sizing, "")
    return "$" + latex.replace(r"\text{", r"\mathrm{") + "$"


def _units(latex: str) -> str | None:
    """DESC's units (``"A \\cdot m^{-2}"``, ``"~"`` for none) as a mathtext string, or ``None``."""
    latex = latex.strip()
    if latex in ("", "~"):
        return None
    return "$\\mathrm{" + latex + "}$"


def _grid(eq, rho, theta, zeta, sfl):
    """The DESC grid over ``rho``, ``theta``, ``zeta`` (logical or PEST angles)."""
    nfp = int(eq.NFP)
    if sfl is None:
        from desc.grid import LinearGrid

        # a LinearGrid may only span one field period; the full torus needs NFP=1
        one_period = zeta.max() <= 2 * np.pi / nfp * (1 + 1e-12)
        return LinearGrid(rho=rho, theta=theta, zeta=zeta, NFP=nfp if one_period else 1)
    from desc.grid import Grid

    # DESC's coordinates of the PEST points, in DESC's node order (zeta slowest, theta fastest)
    z, r, t = np.meshgrid(zeta, rho, theta, indexing="ij")
    points = np.column_stack([r.ravel(), t.ravel(), z.ravel()])
    nodes = eq.map_coordinates(points, inbasis=("rho", "theta_PEST", "zeta"), period=(np.inf, 2 * np.pi, np.inf))
    return Grid(np.asarray(nodes), sort=False, jitable=False)


def from_desc(
    eq,
    names: str | Sequence[str],
    *,
    rho: int | float | Sequence[float] = 11,
    theta: int | float | Sequence[float] = 32,
    zeta: int | float | Sequence[float] = 24,
    sfl: str | None = None,
) -> xr.Dataset:
    """Evaluate DESC quantities on a grid, as a Dataset in plasma-plots' conventions.

    See :mod:`plasma_plots.desc` for what the Dataset contains.

    Parameters
    ----------
    eq : desc.equilibrium.Equilibrium
        The equilibrium, e.g. ``desc.io.load("eq.h5")`` or ``desc.examples.get("W7-X")``.
    names : str or sequence of str
        DESC's names of the quantities (``"|B|"``, ``"iota"``, ``"sqrt(g)"``, ``"B"``, ``"J"``,
        ``"p"``, ``"D_Mercier"``, ``"V"``, ...; see DESC's list of variables). ``"theta_PEST"``
        becomes the coordinate ``theta_P``. Quantities that are not scalars, profiles, fields or
        3-vectors (such as ``"grad(B)"``) are not supported, nor ``"x"``: the position is ``X``,
        ``Y``, ``Z``.
    rho : int, float or sequence of float, optional
        The flux surfaces: an integer as that many points from 0 to 1, a float or a sequence as
        the values. Default: 11 points.
    theta : int, float or sequence of float, optional
        The poloidal angles (PEST angles with ``sfl="pest"``): an integer as that many points
        over ``[0, 2π)``, else the values. Default: 32 points.
    zeta : int, float or sequence of float, optional
        The toroidal angles: an integer as that many points over one field period
        ``[0, 2π/nfp)``, else the values, which may cover the whole torus. Default: 24 points.
    sfl : {None, "pest"}, optional
        ``"pest"`` for a grid in the straight-field-line PEST angle ``theta_P`` (DESC's own
        ``theta`` becomes a coordinate, found by ``eq.map_coordinates``) rather than in DESC's poloidal
        angle. Default: ``None``.

    Returns
    -------
    xarray.Dataset
        The quantities over ``rho``, ``theta`` (or ``theta_P``) and ``zeta``, vectors along
        ``component``, with the coordinates ``X``, ``Y``, ``Z``, labels, units, the angles'
        ``period`` and the ``nfp`` attribute.

    Raises
    ------
    ValueError
        For an unknown ``sfl``, a name DESC doesn't know, a quantity of an unsupported shape or a
        coordinate triplet such as ``"x"``.

    Examples
    --------
    >>> ev = from_desc(eq, ["|B|", "iota", "sqrt(g)"], rho=11, theta=64, zeta=40)
    >>> ev["|B|"].plasma.plot.panels(sweep="zeta", coords="physical", plane="RZ", nrows=1, ncols=3)
    >>> pest = from_desc(eq, "|B|", rho=[0.5], theta=64, zeta=48, sfl="pest")
    >>> pest["|B|"].plasma.plot.slice(x="zeta", y="theta_P", rho=0.5)
    """
    from desc.compute import data_index

    if sfl not in POLOIDAL:
        raise ValueError(f"sfl must be None or 'pest', got {sfl!r}")
    names = [names] if isinstance(names, str) else list(names)
    index = data_index["desc.equilibrium.equilibrium.Equilibrium"]
    unknown = [name for name in names if name not in index]
    if unknown:
        raise ValueError(f"DESC has no quantities {unknown}")
    unsupported = [name for name in names if index[name]["dim"] not in (0, 1, 3)]
    if unsupported:
        raise ValueError(f"quantities of shapes other than scalar or 3-vector are not supported: {unsupported}")
    triplets = [name for name in names if index[name]["description"].startswith("Coordinate triplet")]
    if triplets:
        raise ValueError(f"{triplets} are coordinates, not vectors: the position is the coordinates X, Y, Z")

    nfp = int(eq.NFP)
    rho = _points(rho, 1.0, endpoint=True)
    theta = _points(theta, 2 * np.pi, endpoint=False)
    zeta = _points(zeta, 2 * np.pi / nfp, endpoint=False)
    shape = (zeta.size, rho.size, theta.size)  # DESC's node order: zeta slowest, theta fastest
    grid = _grid(eq, rho, theta, zeta, sfl)
    wanted = list(dict.fromkeys([*names, *GEOMETRY, *(["theta", "theta_PEST"] if sfl == "pest" else [])]))
    with warnings.catch_warnings():  # a full-torus grid has NFP=1, the equilibrium's basis not
        warnings.filterwarnings("ignore", message="Unequal number of field periods")
        values = eq.compute(wanted, grid=grid)

    def field(value) -> np.ndarray:
        """Node values as (rho, theta, zeta), and a vector's components first."""
        value = np.asarray(value, dtype=float)
        return np.moveaxis(value.reshape(*shape, *value.shape[1:]), (0, 1, 2), (-1, -3, -2))

    if not (np.allclose(field(values["rho"]), rho[:, None, None]) and
            np.allclose(field(values["zeta"]), zeta[None, None, :])):  # pragma: no cover - a DESC change
        raise RuntimeError("DESC's grid nodes are not in the expected order")
    if sfl == "pest":
        missed = np.angle(np.exp(1j * (field(values["theta_PEST"]) - theta[None, :, None])))
        if not np.all(np.abs(missed) < 1e-4):
            raise RuntimeError(f"DESC did not find the PEST angles (off by up to {np.nanmax(np.abs(missed)):.2g})")

    dims = ("rho", POLOIDAL[sfl], "zeta")
    coords = {
        "rho": ("rho", rho, {"label": r"$\rho$", "long_name": "normalized toroidal flux radius"}),
        dims[1]: (dims[1], theta, {"label": r"$\theta$" if sfl is None else r"$\theta_P$",
                                   "long_name": "poloidal angle" if sfl is None else "PEST poloidal angle",
                                   "period": 2 * np.pi}),
        "zeta": ("zeta", zeta, {"label": r"$\zeta$", "long_name": "toroidal angle", "period": 2 * np.pi / nfp}),
        **{axis: (dims, field(values[axis]), {"label": f"${axis}$", "units": "m"}) for axis in "XYZ"},
    }
    if sfl == "pest":
        coords["theta"] = (dims, field(values["theta"]),
                           {"label": r"$\theta$", "long_name": "DESC poloidal angle", "period": 2 * np.pi})
    phi = field(values["phi"])

    variables = {}
    for name in names:
        entry = index[name]
        attrs = {"label": _label(entry["label"]), "long_name": entry["description"], "nfp": nfp}
        if (units := _units(entry["units"])) is not None:
            attrs["units"] = units
        value = values[name]
        if name in GRID_QUANTITIES:
            if GRID_QUANTITIES[name] not in dims:  # on a PEST grid, theta_P is the dimension itself
                coords[GRID_QUANTITIES[name]] = (dims, field(value), {"label": r"$\theta_P$", "period": 2 * np.pi})
        elif entry["dim"] == 0 or np.ndim(value) == 0:
            variables[name] = ((), float(np.squeeze(value)), attrs)
        elif entry["dim"] == 3:
            R, p, Z = field(value)  # DESC's vectors are cylindrical: (R, phi, Z)
            cartesian = np.stack([R * np.cos(phi) - p * np.sin(phi), R * np.sin(phi) + p * np.cos(phi), Z])
            variables[name] = (("component", *dims), cartesian, attrs)
        elif entry["coordinates"] == "r":
            variables[name] = (("rho",), field(value)[:, 0, 0], attrs)
        else:
            variables[name] = (dims, field(value), attrs)
    if any(dims_[:1] == ("component",) for dims_, *_ in variables.values()):
        coords["component"] = ("component", CARTESIAN)
    return xr.Dataset(variables, coords=coords, attrs={"nfp": nfp})
