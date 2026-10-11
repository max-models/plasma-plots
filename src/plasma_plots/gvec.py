"""Read GVEC's evaluation Datasets in plasma-plots' conventions.

`GVEC <https://gvec.readthedocs.io>`_ evaluates an equilibrium into ``xarray`` Datasets
(``state.evaluate(...)``, ``state.evaluate_sfl(...)``, ``gvec.Evaluations``) with the dimensions
``rad``, ``pol``, ``tor``, the flux coordinates ``rho``, ``theta``, ``zeta`` (or the Boozer angles
``theta_B``, ``zeta_B``, or the PEST angle ``theta_P``) over them, vectors along ``xyz``, the
position as the variable ``pos`` and LaTeX ``symbol`` attributes. :func:`from_gvec` returns the
same data the way the rest of plasma-plots reads it:

* the dimensions are the flux coordinates themselves: ``rho``, ``theta``/``theta_B``/``theta_P``
  and ``zeta``/``zeta_B``, so ``rho=0.5`` or ``zeta=0`` selects and ``x="zeta"`` draws;
* ``pos`` becomes the physical coordinates ``X``, ``Y``, ``Z`` of every variable (for
  ``coords="physical"``, 3-D views and the vector calculus on the mapped domain), and ``X1``,
  ``X2`` (GVEC's reference coordinates, ``R`` and ``Z`` for the cylindrical map), ``theta_P`` and
  the logical angles of a Boozer or PEST grid become coordinates as well;
* ``xyz`` becomes ``component``;
* each ``symbol`` becomes a ``label`` (``$\\iota$``), which plots use for axes and color bars;
* the angles get a ``period`` attribute (2π, and 2π/nfp toroidally), from which the mode
  spectra take their periods and full-torus mode numbers, and the ``coordinate_lines`` overlay
  its lines;
* GVEC's quadrature weights (``rad_weight``, ``pol_weight``, ``tor_weight``) become
  ``rho_weight``, ``theta_weight``, ``zeta_weight``, which integrals use. GVEC's toroidal weight
  counts every field period; ``zeta_weight`` is divided by nfp, so that an integral covers the
  sampled grid (one field period) as on every other grid: multiply by nfp for the device;
* the number of field periods (``N_FP``) is the ``nfp`` attribute of the Dataset and of every
  variable.

The ``.plasma`` accessor applies :func:`from_gvec` by itself to GVEC data, so
``ev.mod_B.plasma.plot.slice(x="zeta", y="theta", rho=1.0)`` needs no call. Call it once on the
whole Dataset to attach the geometry to every variable first: a single variable such as
``ev.mod_B`` doesn't carry ``pos``. GVEC itself is not needed.

Examples
--------
>>> import plasma_plots
>>> ev = plasma_plots.from_gvec(
...     state.evaluate(
...         "mod_B", "pos", "iota", "N_FP", rho=11, theta=32, zeta=24
...     )
... )
>>> ev.mod_B.plasma.plot.slice(
...     coords="physical",
...     plane="RZ",
...     zeta=0.0,
...     overlays={"coordinate_lines": {"rho": 5, "theta": 8}},
... )
>>> ev.iota.plasma.plot.lineout(rationals=4)
"""

from __future__ import annotations

import numpy as np
import xarray as xr

GVEC_DIMS = ("rad", "pol", "tor")
#: For each GVEC dimension, the 1-D coordinates that can name it, in order of preference.
AXES = {
    "rad": ("rho",),
    "pol": ("theta_B", "theta_P", "theta"),
    "tor": ("zeta_B", "zeta"),
}
#: Variables that describe the grid rather than a field; they become coordinates.
GRID_VARIABLES = ("X1", "X2", "theta_P", "theta", "zeta", "theta_B", "zeta_B")
POLOIDAL = ("theta", "theta_B", "theta_P")
TOROIDAL = ("zeta", "zeta_B")


def _label(symbol: str) -> str:
    """A GVEC ``symbol`` as a mathtext label that Matplotlib and VTK both render: without
    ``\\left``/``\\right`` (``|B|`` rather than ``\\left|B\\right|``) and ``\\text`` as ``\\mathrm``.
    """
    for sizing in (r"\left", r"\right"):
        symbol = symbol.replace(sizing, "")
    return "$" + symbol.replace(r"\text{", r"\mathrm{") + "$"


def is_gvec(data: xr.DataArray | xr.Dataset) -> bool:
    """Whether ``data`` is still in GVEC's layout.

    That is a ``rad``, ``pol`` or ``tor`` dimension with one of GVEC's flux coordinates on it.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The data.

    Returns
    -------
    bool
        ``True`` for GVEC's evaluations (before :func:`from_gvec`), ``False`` otherwise.
    """
    gvec_dims = [d for d in GVEC_DIMS if d in data.dims]
    return any(name in data.coords and data.coords[name].dims == (dim,) for dim in gvec_dims for name in AXES[dim])


def _nfp(ds: xr.Dataset, nfp):
    if nfp is not None:
        return int(nfp)
    if "N_FP" in ds.variables and ds["N_FP"].ndim == 0:
        return int(ds["N_FP"])
    if "nfp" in ds.attrs:
        return int(ds.attrs["nfp"])
    for name in TOROIDAL:  # a uniform grid over one field period, as gvec.Evaluations makes by default
        if name in ds.coords and ds.coords[name].ndim == 1 and ds.coords[name].size > 2:
            zeta = np.asarray(ds.coords[name], dtype=float)
            step = np.diff(zeta)
            if np.allclose(step, step[0]) and np.isclose(zeta[0], 0.0):
                periods = 2 * np.pi / (zeta.size * step[0])
                if periods > 0.5 and np.isclose(periods, round(periods), rtol=1e-6):
                    return int(round(periods))
    return None


def from_gvec(data: xr.DataArray | xr.Dataset, *, nfp: int | None = None) -> xr.DataArray | xr.Dataset:
    """Return GVEC evaluations in plasma-plots' conventions; see :mod:`plasma_plots.gvec`.

    Data that is not in GVEC's layout (see :func:`is_gvec`) comes back with only the steps that
    apply, so calling it twice is harmless.

    Parameters
    ----------
    data : xarray.Dataset or xarray.DataArray
        A Dataset from ``state.evaluate(...)``, ``state.evaluate_sfl(...)`` or ``gvec.Evaluations``
        (also after ``to_netcdf`` and ``open_dataset``), or one of its variables.
    nfp : int, optional
        The number of field periods. Default: the ``N_FP`` variable (evaluate ``"N_FP"`` with the
        rest), else an ``nfp`` attribute, else from a toroidal grid that spans one field period
        uniformly (GVEC's default grid); without one, the toroidal angle gets no ``period``.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The same data, of the same type, with the dimensions ``rho``, ``theta``/``theta_B``/
        ``theta_P``, ``zeta``/``zeta_B`` and ``component``, the coordinates ``X``, ``Y``, ``Z`` (from
        ``pos``), ``X1``, ``X2``, labels from the ``symbol`` attributes, the angles' ``period`` and
        the ``nfp`` attribute.

    Examples
    --------
    >>> ev = from_gvec(
    ...     state.evaluate("mod_B", "pos", "N_FP", rho=11, theta=32, zeta=24)
    ... )
    >>> ev.mod_B.plasma.plot.panels(
    ...     sweep="zeta", coords="physical", plane="RZ", nrows=1, ncols=3
    ... )
    >>> boozer = from_gvec(
    ...     state.evaluate_sfl(
    ...         "mod_B", "pos", rho=[0.5], theta=32, zeta=24, sfl="boozer"
    ...     )
    ... )
    >>> boozer.mod_B.plasma.plot.slice(x="zeta_B", y="theta_B", rho=0.5)
    """
    if isinstance(data, xr.DataArray):
        name = data.name if data.name is not None else "__gvec_value__"
        converted = from_gvec(data.to_dataset(name=name), nfp=nfp)[name]
        return converted.rename(data.name)
    ds = data.copy()
    nfp = _nfp(ds, nfp)

    # --- the grid's own variables become coordinates of every field --- #
    grid = [n for n in GRID_VARIABLES if n in ds.data_vars and set(ds[n].dims) & set(GVEC_DIMS)]
    ds = ds.set_coords(grid)
    if "pos" in ds.data_vars and "xyz" in ds["pos"].dims:
        pos = ds["pos"]
        ds = ds.assign_coords(
            {axis: pos.sel(xyz=component, drop=True) for axis, component in zip("XYZ", "xyz")}
        ).drop_vars("pos")
        for axis in "XYZ":
            ds[axis].attrs = {"label": f"${axis}$"}

    # --- the flux coordinates become the dimensions --- #
    swaps = {}
    for dim, names in AXES.items():
        if dim not in ds.dims:
            continue
        for name in names:
            if name in ds.coords and ds.coords[name].dims == (dim,):
                swaps[dim] = name
                break
    indexed = [name for name in swaps.values() if name in ds.xindexes]
    if indexed:
        ds = ds.drop_indexes(indexed)
    ds = ds.swap_dims(swaps)
    for dim, name in swaps.items():
        weight = f"{dim}_weight"
        if weight in ds.coords:
            ds = ds.rename_vars({weight: f"{name}_weight"})
            if dim == "tor" and nfp is not None:  # GVEC's weight integrates over all field periods
                ds = ds.assign_coords({f"{name}_weight": ds[f"{name}_weight"] / nfp})
    if "xyz" in ds.dims:
        ds = ds.rename({"xyz": "component"})

    # --- labels, periods and nfp --- #
    for name in (*ds.coords, *ds.data_vars):
        attrs = ds[name].attrs
        if "symbol" in attrs and "label" not in attrs:
            attrs["label"] = _label(attrs["symbol"])
        if name in POLOIDAL:
            attrs.setdefault("period", 2 * np.pi)
        elif name in TOROIDAL and nfp is not None:
            attrs.setdefault("period", 2 * np.pi / nfp)
    if nfp is not None:
        ds.attrs["nfp"] = nfp
        for name in ds.data_vars:
            ds[name].attrs.setdefault("nfp", nfp)
    return ds
