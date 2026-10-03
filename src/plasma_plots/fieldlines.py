"""Magnetic field lines on the mapped grid, traced, cut, classified and sampled along.

A field line follows the direction of a vector field, ``dx/ds = B/|B|`` with ``s`` the arc length.
Here it is traced in the logical coordinates of the grid (``eta1``, ``eta2``, ``eta3``, or GVEC's
``rho``, ``theta``, ``zeta``; see :func:`plasma_plots.arrays.logical_dims`), where the grid is
rectangular and the angles wrap around: ``dη/ds = J⁻¹ B / |B|``, with the Jacobian ``J`` of the
mapping from the ``X``, ``Y``, ``Z`` coordinates. The contravariant unit field is interpolated
trilinearly between the grid points and integrated with a fourth-order Runge-Kutta scheme in fixed
steps of arc length. A line stops when it leaves the grid through a bounded direction (the radius,
or the ends of a torus sector), after a number of toroidal transits, or after a length.

The result is a Dataset over ``(s, line)``, like an orbits product is over ``(t, marker)``: the
logical coordinates, the positions ``x``, ``y``, ``z`` and ``|B|`` along each line, and per line
its rotational transform, its toroidal transits, where it ended and its connection length. The
punctures of a poloidal section are recorded while tracing, at every step, so a Poincaré plot does
not depend on how densely the line is saved. The other functions work on that Dataset:
:func:`poincare_section`, :func:`classify_field_lines` and :func:`islands`, :func:`footprint` and
:func:`seed_grid`, :func:`sample_along` and :func:`parallel_wavenumber`;
:mod:`plasma_plots.fieldline_plots` draws them.

Trilinear interpolation of the direction field is second-order accurate in the grid spacing; a
Poincaré plot on a coarse grid shows that as a slow drift of the punctures, not as islands with a
rational transform. Refine the evaluation grid, not the step, when that matters.

Examples
--------
>>> lines = trace_field_lines(B.isel(t=-1), seeds=12, turns=50)
>>> lines.plasma.plot.poincare()  # R, Z punctures of the plane zeta = 0
>>> lines.iota  # the rotational transform of each line
>>> # the mode along one field line, and its parallel wavenumber
>>> along = sample_along(phi, lines)
>>> parallel_wavenumber(along)
"""

from __future__ import annotations

from math import gcd

import numpy as np
import xarray as xr

from .arrays import (
    angle_period,
    logical_dims,
    mapping_jacobian,
    periodicity,
    validate_array,
)

#: The classes of :func:`classify_field_lines`: on a flux surface, inside an island, or chaotic.
LINE_CLASSES = {0: "surface", 1: "island", 2: "chaotic"}


def _label(data):
    return data.attrs.get("label") or data.attrs.get("long_name") or data.name or ""


def _provenance(data) -> dict:
    return {
        key: value for key, value in data.attrs.items() if key in ("run", "run_name")
    }


def _fold(delta, period):
    """Differences of an angle folded into ``(-period/2, period/2]``."""
    return (delta + period / 2) % period - period / 2


# ---------------------------------------------------------------------------------------------
# The grid: axes, periods, trilinear interpolation
# ---------------------------------------------------------------------------------------------
class _Grid:
    """The logical grid of a field: its axes, which of them wrap around, and trilinear interpolation.

    Values are interpolated at points ``eta`` of shape ``(n, 3)``; around a periodic direction the
    cell between the last grid point and the first one (one period on) is interpolated too, and a
    point outside a bounded direction is reported by :meth:`inside`.
    """

    def __init__(self, data: xr.DataArray):
        self.dims = logical_dims(data)
        missing = [d for d in self.dims if d not in data.dims]
        if missing:
            raise ValueError(
                f"field lines need the logical dimensions {self.dims}; {missing} are missing"
            )
        if any(name not in data.coords for name in ("X", "Y", "Z")):
            raise ValueError("field lines need the X, Y, Z coordinates of the grid")
        self.axes = [np.asarray(data[d], dtype=float) for d in self.dims]
        self.points = np.stack(
            [
                np.asarray(data.coords[n].transpose(*self.dims), dtype=float)
                for n in ("X", "Y", "Z")
            ],
            axis=-1,
        )
        self.periods: list[float | None] = []
        self.kinds: list[str | None] = []
        self.angle: list[bool] = []
        for i, dim in enumerate(self.dims):
            kind = periodicity(self.points, i) if self.axes[i].size >= 3 else None
            period = angle_period(data, dim)
            self.angle.append(period is not None)
            if period is None and kind is not None:
                step = self.axes[i][1] - self.axes[i][0]
                period = step * (self.axes[i].size - (1 if kind == "closed" else 0))
            self.kinds.append(kind)
            self.periods.append(period)
        # the grid without repeated endpoints, which interpolation works on
        self.unique_axes = [self.strip(axis, i, 0) for i, axis in enumerate(self.axes)]
        self.points = self.strip(self.points)
        self.uniform = []
        for axis in self.unique_axes:
            h = np.diff(axis)
            self.uniform.append(
                h.size > 0 and bool(np.allclose(h, h[0], rtol=1e-6, atol=0))
            )
        self.nfp = int(data.attrs.get("nfp", 1) or 1)
        self.radial_step = (
            float(np.median(np.diff(self.axes[0]))) if self.axes[0].size > 1 else 0.0
        )

    def strip(
        self, values: np.ndarray, direction: int | None = None, axis: int | None = None
    ) -> np.ndarray:
        """``values`` without the repeated endpoint of each closed direction: of one logical
        ``direction`` (its array ``axis``), or of all three, the last three axes of ``values``.
        """
        if direction is not None:
            if self.kinds[direction] == "closed":
                return np.take(values, range(values.shape[axis] - 1), axis=axis)
            return values
        for d in range(3):
            values = self.strip(values, d, values.ndim - 3 + d)
        return values

    def wrap(self, eta: np.ndarray) -> np.ndarray:
        """Angles folded into ``[x0, x0 + period)``."""
        out = np.array(eta, dtype=float, copy=True)
        for i, period in enumerate(self.periods):
            if period is not None:
                x0 = self.axes[i][0]
                out[..., i] = (out[..., i] - x0) % period + x0
        return out

    def inside(self, eta: np.ndarray) -> np.ndarray:
        """Whether each point lies within every bounded direction."""
        ok = np.ones(eta.shape[:-1], dtype=bool)
        for i, period in enumerate(self.periods):
            if period is None:
                lo, hi = self.axes[i][0], self.axes[i][-1]
                ok &= (eta[..., i] >= lo) & (eta[..., i] <= hi)
        return ok

    def interpolate(self, values: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of ``values`` (shape ``(..., n1, n2, n3)``, without repeated
        endpoints, see :meth:`strip`) at the points ``eta`` of shape ``(n, 3)``."""
        sizes = [values.shape[-3 + a] for a in range(3)]
        indices, fractions = [], []
        for i, period in enumerate(self.periods):
            axis = self.unique_axes[i]
            n = sizes[i]
            x = eta[..., i]
            if n == 1:
                indices.append(np.zeros(x.shape, dtype=int))
                fractions.append(np.zeros(x.shape))
                continue
            if period is not None:
                x = (x - axis[0]) % period + axis[0]
            last = n - 1 if period is not None else n - 2
            if self.uniform[i]:
                h = axis[1] - axis[0]
                k = np.clip(np.floor((x - axis[0]) / h).astype(int), 0, last)
                fractions.append((x - axis[0]) / h - k)
            else:
                edges = (
                    np.append(axis, axis[0] + period) if period is not None else axis
                )
                k = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, last)
                fractions.append((x - edges[k]) / (edges[k + 1] - edges[k]))
            indices.append(k)
        (i, j, k), (fi, fj, fk) = indices, fractions
        out = 0.0
        for di in (0, 1):
            wi = fi if di else 1 - fi
            ii = (i + di) % sizes[0]
            for dj in (0, 1):
                wj = fj if dj else 1 - fj
                jj = (j + dj) % sizes[1]
                for dk in (0, 1):
                    wk = fk if dk else 1 - fk
                    kk = (k + dk) % sizes[2]
                    out = out + values[..., ii, jj, kk] * (wi * wj * wk)
        return out

    def positions(self, eta: np.ndarray) -> np.ndarray:
        """``X``, ``Y``, ``Z`` at points ``(..., 3)``, as ``(3, ...)``; NaN where a point is NaN."""
        flat = eta.reshape(-1, 3)
        finite = np.isfinite(flat).all(axis=1)
        out = np.full((3, flat.shape[0]), np.nan)
        if finite.any():
            out[:, finite] = self.interpolate(
                np.moveaxis(self.points, -1, 0), flat[finite]
            )
        return out.reshape(3, *eta.shape[:-1])

    def values_at(self, values: np.ndarray, eta: np.ndarray) -> np.ndarray:
        """``values`` ``(..., n1, n2, n3)`` at points ``(..., 3)``, NaN where a point is NaN."""
        flat = eta.reshape(-1, 3)
        finite = np.isfinite(flat).all(axis=1)
        out = np.full((*values.shape[:-3], flat.shape[0]), np.nan)
        if finite.any():
            out[..., finite] = self.interpolate(values, flat[finite])
        return out.reshape(*values.shape[:-3], *eta.shape[:-1])


def _direction_field(data: xr.DataArray, components: str):
    """``(b, absB, grid)``: the contravariant unit field ``(3, n1, n2, n3)``, ``|B|`` and the grid."""
    validate_array(data)
    if "component" not in data.dims or data.sizes["component"] != 3:
        raise ValueError(
            "field lines need a vector field with a 'component' dimension of size 3"
        )
    if components not in ("cartesian", "contravariant"):
        raise ValueError(
            f'components must be "cartesian" or "contravariant"; got {components!r}'
        )
    grid = _Grid(data)
    extra = [d for d in data.dims if d not in ("component", *grid.dims)]
    if extra:
        raise ValueError(
            f"select the dimensions {extra} first (e.g. {extra[0]}=-1); field lines are traced "
            "through one snapshot of the field"
        )
    short = [d for d in grid.dims if data.sizes[d] < 2]
    if short:
        raise ValueError(
            f"field lines need at least two points along each of {grid.dims}; {short} have fewer"
        )
    field = np.asarray(data.transpose("component", *grid.dims), dtype=float)
    jacobian = mapping_jacobian(data.isel(component=0, drop=True).transpose(*grid.dims))
    if components == "contravariant":
        cartesian = np.einsum("ai...,i...->a...", jacobian, field)
        contravariant = field
    else:
        cartesian = field
        inverse = np.linalg.pinv(np.moveaxis(jacobian, (0, 1), (-2, -1)))
        contravariant = np.einsum("...ia,a...->i...", inverse, field)
    absB = np.linalg.norm(cartesian, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(absB > 0, contravariant / absB, 0.0)
    return grid.strip(unit), grid.strip(absB), grid


def _seed_points(grid: _Grid, seeds) -> np.ndarray:
    """The seeds as logical points ``(n, 3)`` in the order of the logical dimensions."""
    if isinstance(seeds, (int, np.integer)):
        radial = grid.axes[0]
        span = radial[-1] - radial[0]
        values = np.linspace(
            radial[0] + 0.05 * span, radial[-1] - 0.05 * span, int(seeds)
        )
        return np.column_stack(
            [
                values,
                np.full(values.size, grid.axes[1][0]),
                np.full(values.size, grid.axes[2][0]),
            ]
        )
    if isinstance(seeds, dict):
        unknown = set(seeds) - set(grid.dims)
        if unknown:
            raise ValueError(
                f"unknown seed coordinates {sorted(unknown)}; the logical dimensions are {grid.dims}"
            )
        arrays = [
            np.atleast_1d(np.asarray(seeds.get(dim, grid.axes[i][0]), dtype=float))
            for i, dim in enumerate(grid.dims)
        ]
        mesh = np.meshgrid(*arrays, indexing="ij")
        return np.column_stack([m.ravel() for m in mesh])
    if isinstance(seeds, xr.Dataset):
        missing = [dim for dim in grid.dims if dim not in seeds]
        if missing:
            raise ValueError(f"a seeds Dataset needs the variables {missing}")
        return np.column_stack(
            [np.asarray(seeds[dim], dtype=float).ravel() for dim in grid.dims]
        )
    points = np.asarray(seeds, dtype=float)
    if points.ndim == 1 and points.size == 3:
        points = points[None]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "seeds must be a number of lines, a dict of logical coordinates, or an (n, 3) array of "
            f"logical points; got shape {points.shape}"
        )
    return points


def _default_step(grid: _Grid) -> float:
    """The median spacing of the grid points in physical space."""
    spacings = []
    for axis in range(3):
        if grid.points.shape[axis] > 1:
            d = np.linalg.norm(np.diff(grid.points, axis=axis), axis=-1)
            spacings.append(d[d > 0])
    return float(np.median(np.concatenate(spacings)))


def _toroidal_turn(grid: _Grid) -> float | None:
    """The toroidal coordinate of one turn of the torus: nfp periods of the toroidal logical
    coordinate (GVEC's ``period`` is one field period; Struphy's ``eta3`` has nfp 1)."""
    period = grid.periods[2]
    return None if period is None else period * grid.nfp


# ---------------------------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------------------------
def trace_field_lines(
    data: xr.DataArray,
    *,
    seeds=8,
    turns: float | None = None,
    length: float | None = None,
    step: float | None = None,
    direction: str = "forward",
    section: float | None = None,
    stride: int | None = None,
    components: str = "cartesian",
    max_steps: int = 2_000_000,
) -> xr.Dataset:
    """Trace field lines of a vector field through its mapped grid.

    See :mod:`plasma_plots.fieldlines` for the method. Each line starts at a seed and stops when it
    has made ``turns`` toroidal transits, has reached the arc ``length``, or has left the grid
    through a bounded direction (the radius, or the ends of a torus sector); afterwards its samples
    are NaN. The punctures of the poloidal plane ``section`` (a value of the toroidal logical
    coordinate) are recorded at every step while tracing, for :func:`poincare_section`.

    Parameters
    ----------
    data : xarray.DataArray
        The vector field, with a ``component`` dimension of size 3 and the three logical
        dimensions (every other dimension selected, e.g. ``B.isel(t=-1)``), with the ``X``, ``Y``,
        ``Z`` coordinates of the grid.
    seeds : int, dict, array_like or xarray.Dataset, optional
        Where the lines start, in logical coordinates: a number of seeds spread along the radial
        coordinate at the first poloidal and toroidal grid values (the outboard midplane of a torus
        whose angles start at 0); a dict of logical coordinates to values or 1-D arrays, which are
        combined into a grid of seeds (e.g. ``{"rho": 0.95, "theta": thetas, "zeta": zetas}`` for a
        connection-length map; a missing coordinate takes its first grid value); an ``(n, 3)``
        array of points in the order of the logical dimensions; or a Dataset with one variable per
        logical coordinate. Default: 8.
    turns : float, optional
        Stop a line after this many toroidal transits (turns of the torus: nfp periods of the
        toroidal logical coordinate, i.e. 2π of GVEC's toroidal angle, one period of Struphy's
        ``eta3``). Default: 20 when the toroidal direction wraps around, else none.
    length : float, optional
        Stop a line after this arc length, in the units of ``X``, ``Y``, ``Z``. Default: with
        ``turns``, 1.5 times the length ``turns`` circles through the seeds' mean major radius
        would take (a safety net); without, four times the grid's extent.
    step : float, optional
        The step of arc length of the integrator. Default: the median spacing of the grid
        points (trilinear interpolation limits the accuracy before the step does).
    direction : {"forward", "backward", "both"}, optional
        Along the field, against it, or each seed in both directions (two lines per seed, with
        the coordinates ``seed`` and ``direction`` telling them apart; the connection length is
        then the sum of both). Default: ``"forward"``.
    section : float, optional
        The toroidal logical coordinate of the poloidal plane whose punctures are recorded.
        Default: the first toroidal grid value (e.g. ``zeta = 0``).
    stride : int, optional
        Save every ``stride``-th step of each line. Default: as many as keep each line at about
        20 000 samples at most; the punctures, transits and lengths count every step regardless.
    components : {"cartesian", "contravariant"}, optional
        What the components are: ``"cartesian"`` (default) ``(x, y, z)``, or contravariant
        logical components, as for :func:`plasma_plots.analysis.divergence`.
    max_steps : int, optional
        A cap on the number of steps of each line. Default: 2 000 000.

    Returns
    -------
    xarray.Dataset
        Over ``(s, line)``: the logical coordinates of each line (angles folded into one period),
        its positions ``x``, ``y``, ``z`` and ``absB`` (``|B|``), NaN after a line has stopped;
        over ``line``: ``iota`` (poloidal per toroidal turns, from the whole traced line; NaN
        without a toroidal transit), ``transits``, ``length`` (the arc length traced),
        ``exited`` (whether it left the grid), ``connection_length`` (the arc length to the exit,
        summed over both directions with ``direction="both"``; NaN while a line has not left the
        grid), and where each line ended, ``<dim>_end`` and ``x_end``, ``y_end``, ``z_end``; over
        ``(puncture, line)``: ``puncture_<radial>``, ``puncture_<poloidal>``, ``puncture_x``,
        ``puncture_y``, ``puncture_z`` and ``puncture_s``, where each line crosses ``section``
        (padded with NaN). The coordinates ``<dim>_start``, ``seed`` and ``direction`` describe
        the seeds; the attrs hold the section, the dimension names, the periods and the toroidal
        turn.

    Raises
    ------
    ValueError
        If ``data`` is not a three-component field over the three logical dimensions with ``X``,
        ``Y``, ``Z``, has other dimensions left, or the seeds or ``direction`` are malformed.

    See Also
    --------
    poincare_section : The punctures as a Dataset of their own.
    footprint : Where the lines that left the grid did so.
    sample_along : A field along the lines.
    plasma_plots.fieldline_plots.plot_poincare : The Poincaré plot.

    Examples
    --------
    >>> lines = trace_field_lines(B.isel(t=-1), seeds=12, turns=100)
    >>> lines.iota.values  # the rotational transform of each line
    >>> # connection lengths from a surface near the edge, both ways
    >>> edge = trace_field_lines(
    ...     B.isel(t=-1),
    ...     seeds={"eta1": 0.98, "eta2": np.linspace(0, 1, 32), "eta3": 0.0},
    ...     direction="both",
    ...     turns=50,
    ... )
    """
    if direction not in ("forward", "backward", "both"):
        raise ValueError(
            f'direction must be "forward", "backward" or "both"; got {direction!r}'
        )
    unit, absB, grid = _direction_field(data, components)
    starts = _seed_points(grid, seeds)
    if not grid.inside(starts).all():
        raise ValueError("every seed must lie inside the grid's bounded directions")
    n_seeds = starts.shape[0]
    signs = {"forward": [1.0], "backward": [-1.0], "both": [1.0, -1.0]}[direction]
    eta0 = np.concatenate([starts] * len(signs))
    sign = np.repeat(signs, n_seeds)
    seed_index = np.tile(np.arange(n_seeds), len(signs))
    n_lines = eta0.shape[0]

    turn = _toroidal_turn(grid)
    period_pol, period_tor = grid.periods[1], grid.periods[2]
    if turns is None and length is None and turn is not None:
        turns = 20.0
    step = _default_step(grid) if step is None else float(step)
    if step <= 0:
        raise ValueError("step must be positive")
    if length is None:
        if turns is not None and turn is not None:
            xyz = grid.positions(eta0)
            radius = float(np.mean(np.hypot(xyz[0], xyz[1])))
            length = 1.5 * turns * 2 * np.pi * max(radius, step)
        else:
            extent = np.ptp(grid.points.reshape(-1, 3), axis=0)
            length = 4 * float(np.linalg.norm(extent))
    n_steps = max(1, min(int(np.ceil(length / step)), max_steps))
    if stride is None:
        stride = max(1, int(np.ceil(n_steps / 20_000)))
    stride = max(1, int(stride))
    section_value = float(grid.axes[2][0]) if section is None else float(section)

    # --- integrate --- #
    eta = eta0.copy()
    active = np.ones(n_lines, dtype=bool)
    exited = np.zeros(n_lines, dtype=bool)
    traced = np.zeros(n_lines)
    exit_length = np.full(n_lines, np.nan)
    poloidal = np.zeros(n_lines)  # unwrapped advance of the angles
    toroidal = np.zeros(n_lines)
    frames = [eta0.copy()]
    punctures: list[list] = [[] for _ in range(n_lines)]
    for k in range(1, n_steps + 1):
        indices = np.flatnonzero(active)
        if not indices.size:
            break
        current = eta[indices]
        s = sign[indices, None]

        def velocity(points):
            return s * grid.interpolate(unit, points).T  # (n, 3)

        k1 = velocity(current)
        k2 = velocity(current + 0.5 * step * k1)
        k3 = velocity(current + 0.5 * step * k2)
        k4 = velocity(current + step * k3)
        new = current + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        stalled = np.linalg.norm(k1, axis=1) == 0
        inside = grid.inside(new)
        before = traced[indices].copy()
        advanced = np.full(indices.size, step)
        # a line leaving the grid stops at the boundary, interpolated along its step
        for row in np.flatnonzero(~inside):
            fraction = 1.0
            for axis, period in enumerate(grid.periods):
                if period is None:
                    for bound in (grid.axes[axis][0], grid.axes[axis][-1]):
                        d0, d1 = current[row, axis] - bound, new[row, axis] - bound
                        if d0 * d1 < 0:
                            fraction = min(fraction, abs(d0) / (abs(d0) + abs(d1)))
            new[row] = current[row] + fraction * (new[row] - current[row])
            advanced[row] = fraction * step
            exit_length[indices[row]] = before[row] + fraction * step
        exited[indices[~inside]] = True
        traced[indices] = before + advanced
        delta = new - current
        if period_pol is not None:
            delta[:, 1] = _fold(delta[:, 1], period_pol)
        if period_tor is not None:
            delta[:, 2] = _fold(delta[:, 2], period_tor)
        poloidal[indices] += delta[:, 1]
        toroidal[indices] += delta[:, 2]
        # punctures of the section plane
        if period_tor is not None:
            d0 = _fold(current[:, 2] - section_value, period_tor)
            d1 = _fold(new[:, 2] - section_value, period_tor)
            crossing = (d0 * d1 < 0) & (np.abs(d0) + np.abs(d1) < period_tor / 2)
            for row in np.flatnonzero(crossing):
                f = abs(d0[row]) / (abs(d0[row]) + abs(d1[row]))
                point = current[row] + f * (new[row] - current[row])
                point[2] = section_value
                punctures[indices[row]].append((point, before[row] + f * advanced[row]))
        eta[indices] = new
        done = ~inside | stalled
        if turns is not None and turn is not None:
            done |= np.abs(toroidal[indices]) / turn >= turns
        active[indices[done]] = False
        if k % stride == 0:
            frames.append(eta.copy())

    # --- the samples: uniform in s, NaN after a line's end --- #
    samples = np.stack(frames)  # (n_s, line, 3)
    s_values = np.arange(samples.shape[0]) * stride * step
    alive = s_values[:, None] <= traced[None, :] + 1e-9 * step
    samples = np.where(alive[..., None], samples, np.nan)
    wrapped = grid.wrap(samples)
    xyz = grid.positions(wrapped)
    magnitude = grid.values_at(absB, wrapped)
    ends = grid.wrap(eta)
    xyz_end = grid.positions(ends)

    with np.errstate(invalid="ignore", divide="ignore"):
        pol_turns = (
            poloidal / period_pol
            if period_pol is not None
            else np.full(n_lines, np.nan)
        )
        tor_turns = toroidal / turn if turn is not None else np.full(n_lines, np.nan)
        iota = np.where(np.abs(tor_turns) > 0, pol_turns / tor_turns, np.nan)
    connection = exit_length.copy()
    if direction == "both":
        total = exit_length[:n_seeds] + exit_length[n_seeds:]
        connection = np.concatenate([total, total])

    n_punctures = max((len(p) for p in punctures), default=0)
    puncture_eta = np.full((n_punctures, n_lines, 3), np.nan)
    puncture_s = np.full((n_punctures, n_lines), np.nan)
    for i, items in enumerate(punctures):
        for j, (point, where) in enumerate(items):
            puncture_eta[j, i] = point
            puncture_s[j, i] = where
    puncture_eta = grid.wrap(puncture_eta)
    puncture_xyz = grid.positions(puncture_eta)

    radial, pol, tor = grid.dims
    coords = {
        "s": ("s", s_values, {"long_name": "arc length", "label": "$s$"}),
        "line": np.arange(n_lines),
        "seed": ("line", seed_index),
        "direction": ("line", sign.astype(int)),
    }
    for i, dim in enumerate(grid.dims):
        coords[f"{dim}_start"] = ("line", eta0[:, i])
    variables = {}
    for i, dim in enumerate(grid.dims):
        variables[dim] = (("s", "line"), wrapped[..., i], dict(data[dim].attrs))
    for i, name in enumerate(("x", "y", "z")):
        variables[name] = (("s", "line"), xyz[i], {"label": f"${name}$"})
    variables["absB"] = (("s", "line"), magnitude, {"label": "$|B|$"})
    variables["iota"] = (
        ("line",),
        iota,
        {"label": "$\\iota$", "long_name": "rotational transform"},
    )
    variables["transits"] = (
        ("line",),
        np.abs(tor_turns),
        {"long_name": "toroidal transits"},
    )
    variables["length"] = (("line",), traced, {"long_name": "traced arc length"})
    variables["exited"] = (("line",), exited)
    variables["connection_length"] = (
        ("line",),
        connection,
        {"label": "$L_c$", "long_name": "connection length"},
    )
    for i, dim in enumerate(grid.dims):
        variables[f"{dim}_end"] = (("line",), ends[:, i])
    for i, name in enumerate(("x", "y", "z")):
        variables[f"{name}_end"] = (("line",), xyz_end[i])
    variables[f"puncture_{radial}"] = (("puncture", "line"), puncture_eta[..., 0])
    variables[f"puncture_{pol}"] = (("puncture", "line"), puncture_eta[..., 1])
    for i, name in enumerate(("x", "y", "z")):
        variables[f"puncture_{name}"] = (("puncture", "line"), puncture_xyz[i])
    variables["puncture_s"] = (("puncture", "line"), puncture_s)
    out = xr.Dataset(variables, coords=coords)
    out.attrs = {
        **_provenance(data),
        "label": f"field lines of {_label(data)}".strip(),
        "kind": "field_lines",
        "logical_dims": list(grid.dims),
        "section": section_value,
        "step": step,
        "periods": [np.nan if p is None else float(p) for p in grid.periods],
        "toroidal_turn": np.nan if turn is None else float(turn),
        "radial_step": grid.radial_step,
    }
    for name in ("units", "nfp"):
        if name in data.attrs:
            out.attrs[name] = data.attrs[name]
    return out


def _dims_of(lines: xr.Dataset) -> tuple[str, str, str]:
    dims = lines.attrs.get("logical_dims")
    if dims is None:
        raise ValueError(
            "expected the Dataset of trace_field_lines or poincare_section (with logical_dims in its attrs)"
        )
    return tuple(dims)


def _periods_of(lines: xr.Dataset) -> list[float | None]:
    return [
        None if not np.isfinite(p) else float(p)
        for p in lines.attrs.get("periods", [np.nan] * 3)
    ]


def _line_coords(lines: xr.Dataset) -> dict:
    """The per-line coordinates (seeds, direction) and ``line`` itself."""
    return {
        name: lines[name]
        for name in lines.coords
        if name == "line" or lines[name].dims == ("line",)
    }


# ---------------------------------------------------------------------------------------------
# Poincaré sections, rotational transform, islands
# ---------------------------------------------------------------------------------------------
def poincare_section(lines: xr.Dataset, *, angle: float | None = None) -> xr.Dataset:
    """The punctures of a poloidal plane by traced field lines, as a Dataset over ``(puncture, line)``.

    With ``angle`` left out (or equal to the ``section`` the lines were traced with) the punctures
    recorded while tracing are returned, which count every step. Another ``angle`` finds the
    crossings in the saved samples, interpolated linearly between them, which is as fine as the
    lines were saved (see ``stride`` of :func:`trace_field_lines`). A section passed in comes back
    as it is.

    Parameters
    ----------
    lines : xarray.Dataset
        The lines of :func:`trace_field_lines`.
    angle : float, optional
        The toroidal logical coordinate of the plane. Default: the traced section.

    Returns
    -------
    xarray.Dataset
        Over ``(puncture, line)``: the radial and poloidal logical coordinates of each puncture
        (named as in the field), ``x``, ``y``, ``z``, ``R`` (``√(x² + y²)``) and ``s`` (the arc
        length there), padded with NaN; per ``line`` the coordinates ``iota``, ``transits``,
        ``connection_length`` and the seeds. The attrs hold the ``section`` and the dimension
        names.

    Raises
    ------
    ValueError
        If the toroidal direction does not wrap around, so there is no plane to cut, or a section
        is asked to be cut again at another angle.

    See Also
    --------
    classify_field_lines, islands
    plasma_plots.fieldline_plots.plot_poincare : The plot.

    Examples
    --------
    >>> section = poincare_section(lines)
    >>> section.plasma.plot.poincare()
    >>> poincare_section(lines, angle=np.pi / 5).plasma.plot.poincare()
    """
    radial, pol, tor = _dims_of(lines)
    stored = lines.attrs.get("section", np.nan)
    if lines.attrs.get("kind") == "poincare_section":
        if angle is None or np.isclose(angle, stored):
            return lines
        raise ValueError(
            "this is already a Poincaré section; cut the traced lines at another angle instead"
        )
    period = _periods_of(lines)[2]
    if period is None:
        raise ValueError(
            "the toroidal direction does not wrap around; there is no plane to cut"
        )
    if angle is None or np.isclose(angle, stored):
        angle = stored
        values = {
            radial: np.asarray(lines[f"puncture_{radial}"], dtype=float),
            pol: np.asarray(lines[f"puncture_{pol}"], dtype=float),
            **{
                name: np.asarray(lines[f"puncture_{name}"], dtype=float)
                for name in ("x", "y", "z")
            },
            "s": np.asarray(lines["puncture_s"], dtype=float),
        }
    else:
        zeta = np.asarray(lines[tor], dtype=float)
        d = _fold(zeta - angle, period)
        d0, d1 = d[:-1], d[1:]
        crossing = (d0 * d1 < 0) & (np.abs(d0) + np.abs(d1) < period / 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.abs(d0) / (np.abs(d0) + np.abs(d1))
        count = int(crossing.sum(axis=0).max()) if crossing.size else 0
        n_lines = lines.sizes["line"]
        names = (radial, pol, "x", "y", "z", "s")
        arrays = {
            name: np.asarray(
                (
                    lines[name]
                    if name != "s"
                    else lines["s"].broadcast_like(lines[radial])
                ),
                dtype=float,
            )
            for name in names
        }
        period_pol = _periods_of(lines)[1]
        values = {name: np.full((count, n_lines), np.nan) for name in names}
        for i in range(n_lines):
            rows = np.flatnonzero(crossing[:, i])
            f = fraction[rows, i]
            for name in names:
                a = arrays[name]
                d = a[rows + 1, i] - a[rows, i]
                if name == pol and period_pol is not None:
                    d = _fold(d, period_pol)
                values[name][: rows.size, i] = a[rows, i] + f * d
        if period_pol is not None and count:
            x0 = float(np.nanmin(arrays[pol]))
            values[pol] = (values[pol] - x0) % period_pol + x0
    n_punctures = values["s"].shape[0]
    data_vars = {name: (("puncture", "line"), array) for name, array in values.items()}
    data_vars["R"] = (("puncture", "line"), np.hypot(values["x"], values["y"]))
    out = xr.Dataset(
        data_vars, coords={"puncture": np.arange(n_punctures), **_line_coords(lines)}
    )
    for name in ("iota", "transits", "connection_length"):
        if name in lines:
            out = out.assign_coords(
                {name: ("line", np.asarray(lines[name]), dict(lines[name].attrs))}
            )
    out[radial].attrs = dict(lines[radial].attrs)
    out[pol].attrs = dict(lines[pol].attrs)
    out["R"].attrs = {"label": "$R$"}
    for name in ("x", "y", "z"):
        out[name].attrs = {"label": f"${name}$"}
    out["s"].attrs = {"label": "$s$", "long_name": "arc length"}
    out.attrs = {
        **lines.attrs,
        "kind": "poincare_section",
        "section": float(angle),
        "label": f"Poincaré section of {lines.attrs.get('label', 'field lines')}",
    }
    return out


def rotational_transform(lines: xr.Dataset) -> xr.DataArray:
    """The rotational transform of each traced field line, poloidal turns per toroidal turn.

    Computed while tracing from the whole line (its unwrapped poloidal and toroidal advance; a
    toroidal turn is nfp periods of the toroidal coordinate: 2π for GVEC's angles, one period of
    ``eta3`` for Struphy's). NaN for a line that made no toroidal transit.

    Parameters
    ----------
    lines : xarray.Dataset
        The lines of :func:`trace_field_lines`, or a :func:`poincare_section` of them.

    Returns
    -------
    xarray.DataArray
        ``iota`` over ``line``, with the seeds as coordinates.

    Raises
    ------
    ValueError
        If ``lines`` carries no ``iota``.

    See Also
    --------
    plasma_plots.analysis.rational_surfaces : Where a profile of ι is rational.

    Examples
    --------
    >>> iota = rotational_transform(lines)
    >>> iota.swap_dims(line="eta1_start").plasma.plot.lineout()
    """
    if "iota" not in lines.variables:
        raise ValueError(
            "expected the Dataset of trace_field_lines, which carries iota per line"
        )
    out = xr.DataArray(
        np.asarray(lines["iota"], dtype=float),
        dims="line",
        coords=_line_coords(lines),
        name="iota",
        attrs={
            **_provenance(lines),
            "label": "$\\iota$",
            "long_name": "rotational transform",
        },
    )
    return out


def _nearest_rational(iota: float, max_denominator: int, nfp: int, tolerance: float):
    """``(n, m)`` of the lowest-order ``n/m`` within ``tolerance`` of ``iota``, or ``None``."""
    if not np.isfinite(iota):
        return None
    for m in range(1, max_denominator + 1):
        n = int(np.rint(iota * m))
        if nfp > 1 and n % nfp:
            continue
        if (gcd(abs(n), m) == 1 or (n == 0 and m == 1)) and abs(
            iota - n / m
        ) <= tolerance:
            return n, m
    return None


def _folded(section: xr.Dataset):
    """The per-line arrays the island heuristics share."""
    radial, pol, _ = _dims_of(section)
    period = _periods_of(section)[1] or 1.0
    r = np.asarray(section[radial], dtype=float)
    th = np.asarray(section[pol], dtype=float)
    iota = (
        np.asarray(section["iota"], dtype=float)
        if "iota" in section.variables
        else np.full(r.shape[1], np.nan)
    )
    transits = (
        np.asarray(section["transits"], dtype=float)
        if "transits" in section.variables
        else np.isfinite(r).sum(axis=0).astype(float)
    )
    return radial, pol, period, r, th, iota, transits


def classify_field_lines(
    section: xr.Dataset,
    *,
    max_denominator: int = 12,
    tolerance: float | None = None,
    threshold: float = 0.1,
    min_spread: float | None = None,
) -> xr.DataArray:
    """Classify each field line of a Poincaré section: on a flux surface (0), in an island (1) or chaotic (2).

    A heuristic on the punctures of each line, in the order of their arc length. A line whose
    punctures spread less than ``min_spread`` in the radial coordinate lies on a flux surface.
    Otherwise its rotational transform picks the lowest-order rational ``n/m`` within
    ``tolerance`` (``n`` a multiple of nfp when the field has it), and the punctures are folded
    into one island by the angle ``ψ = m θ mod period``. Successive punctures of a line on a
    surface near the rational keep advancing in ``ψ`` (rotation: their accumulated advance grows
    without bound), while those of a line inside an island turn back (libration about the
    O-point): a line is in an island when its accumulated ``ψ`` advance never spans a full
    period. Otherwise the punctures sorted by the poloidal angle should trace a smooth curve, the
    flux surface; when the median radial jump between neighbours in the angle exceeds
    ``threshold`` times the line's radial spread, the line is chaotic. Lines with fewer than six punctures are surfaces. A line on a surface
    whose ι is closer to the rational than one over the number of transits passes as an island;
    trace longer to tell them apart.

    Parameters
    ----------
    section : xarray.Dataset
        A :func:`poincare_section`, or the lines of :func:`trace_field_lines` (cut at their
        section).
    max_denominator : int, optional
        The largest ``m`` of the rationals considered. Default: 12.
    tolerance : float, optional
        How close ι must be to ``n/m``. Default: two over the number of toroidal transits of the
        line (the resolution of ι from the trace), at least 1e-3.
    threshold : float, optional
        The median radial jump between punctures that are neighbours in the poloidal angle,
        relative to the line's radial spread, above which a non-librating line is chaotic (a
        smooth curve through 100 punctures gives a few hundredths). Default: 0.1.
    min_spread : float, optional
        The radial spread of the punctures (in the radial logical coordinate) below which a line
        is on a surface. Default: half the radial grid spacing of the traced field.

    Returns
    -------
    xarray.DataArray
        The class of each line over ``line`` (``flag_meanings`` in the attrs, see
        :data:`LINE_CLASSES`), with the coordinates ``n`` and ``m`` of the rational found (0 when
        none) and ``iota``.

    See Also
    --------
    islands : The island chains and their widths.

    Examples
    --------
    >>> codes = classify_field_lines(poincare_section(lines))
    >>> lines.isel(line=codes.values == 1)  # the lines inside islands
    """
    section = poincare_section(section)
    radial, pol, period, r, th, iota, transits = _folded(section)
    nfp = int(section.attrs.get("nfp", 1) or 1)
    if min_spread is None:
        min_spread = 0.5 * float(section.attrs.get("radial_step", 0.0) or 0.0)
    codes = np.zeros(r.shape[1], dtype=int)
    ns, ms = np.zeros(r.shape[1], dtype=int), np.zeros(r.shape[1], dtype=int)
    for i in range(r.shape[1]):
        keep = np.isfinite(r[:, i]) & np.isfinite(th[:, i])
        tol = (
            tolerance if tolerance is not None else max(2 / max(transits[i], 1.0), 1e-3)
        )
        rational = _nearest_rational(float(iota[i]), max_denominator, nfp, tol)
        if rational:
            ns[i], ms[i] = rational
        if keep.sum() < 6:
            continue
        ri, thi = r[keep, i], th[keep, i]
        spread = float(np.ptp(ri))
        if spread <= min_spread:
            continue
        m = rational[1] if rational else 1
        psi = (m * thi) % period
        if rational:
            advance = np.cumsum(_fold(np.diff(psi), period))
            if (
                np.ptp(advance) < period
            ):  # libration: never a full turn around the island
                codes[i] = 1
                continue
        sorted_r = ri[np.argsort(thi)]
        jumps = np.abs(np.diff(np.append(sorted_r, sorted_r[0])))
        if float(np.median(jumps)) / spread > threshold:
            codes[i] = 2
    return xr.DataArray(
        codes,
        dims="line",
        coords={
            "line": section["line"].values,
            "n": ("line", ns),
            "m": ("line", ms),
            "iota": ("line", iota),
        },
        name="classification",
        attrs={
            **_provenance(section),
            "label": "field line classification",
            "flag_values": list(LINE_CLASSES),
            "flag_meanings": " ".join(LINE_CLASSES.values()),
        },
    )


def islands(
    section: xr.Dataset,
    *,
    max_denominator: int = 12,
    tolerance: float | None = None,
    threshold: float = 0.1,
    min_spread: float | None = None,
) -> xr.Dataset:
    """The island chains a Poincaré section shows, with an estimate of their widths.

    The lines :func:`classify_field_lines` puts inside an island are grouped by their rational
    ``n/m``. The width of a chain is the radial extent of its widest island orbit, the one traced
    closest to the separatrix (the full width of an island, at its O-point), so a chain traced
    only near its O-point is underestimated: seed densely across it. The O-point's poloidal
    angle is where that orbit's punctures are furthest apart radially, folded by ``m``; the
    other O-points are a period over ``m`` apart.

    Parameters
    ----------
    section : xarray.Dataset
        A :func:`poincare_section`, or the lines of :func:`trace_field_lines`.
    max_denominator : int, optional
        The largest ``m`` of the rationals considered. Default: 12.
    tolerance : float, optional
        How close ι must be to ``n/m``; see :func:`classify_field_lines`.
    threshold : float, optional
        The chaos criterion of :func:`classify_field_lines`. Default: 0.1.
    min_spread : float, optional
        The surface criterion of :func:`classify_field_lines`. Default: half the radial grid
        spacing.

    Returns
    -------
    xarray.Dataset
        Over ``chain``: ``n``, ``m``, ``iota`` (``n/m``), ``width`` (in the radial logical
        coordinate), ``width_physical`` (the distance between the orbit's radially outermost and
        innermost punctures in ``x``, ``y``, ``z``), ``center`` (the mean radial coordinate of
        the chain's punctures), ``lines`` (how many lines lie in it) and ``o_point_<poloidal>``
        (the poloidal angle of one O-point). Empty when there is no island.

    See Also
    --------
    classify_field_lines : Which lines are in islands.
    plasma_plots.fieldline_plots.plot_poincare : ``islands=True`` annotates them.

    Examples
    --------
    >>> chains = islands(poincare_section(lines))
    >>> chains.to_dataframe()
    """
    section = poincare_section(section)
    radial, pol, period, r, th, _, _ = _folded(section)
    codes = classify_field_lines(
        section,
        max_denominator=max_denominator,
        tolerance=tolerance,
        threshold=threshold,
        min_spread=min_spread,
    )
    xyz = np.stack([np.asarray(section[n], dtype=float) for n in ("x", "y", "z")], -1)
    chains: dict[tuple[int, int], dict] = {}
    for i in np.flatnonzero(np.asarray(codes) == 1):
        n, m = int(codes.n[i]), int(codes.m[i])
        keep = np.isfinite(r[:, i]) & np.isfinite(th[:, i])
        ri, thi, pi = r[keep, i], th[keep, i], xyz[keep, i]
        lo, hi = int(np.argmin(ri)), int(np.argmax(ri))
        width = float(ri[hi] - ri[lo])
        width_physical = (
            float(np.linalg.norm(pi[hi] - pi[lo]))
            if np.isfinite(pi[[lo, hi]]).all()
            else np.nan
        )
        # the O-point: the angle (folded by m) of the orbit's widest extent
        psi = (m * thi) % period
        o_point = float(((psi[lo] + _fold(psi[hi] - psi[lo], period) / 2) % period) / m)
        entry = chains.setdefault(
            (n, m),
            {
                "width": -1.0,
                "width_physical": np.nan,
                "radii": [],
                "lines": 0,
                "o_point": np.nan,
            },
        )
        if width > entry["width"]:
            entry.update(width=width, width_physical=width_physical, o_point=o_point)
        entry["radii"].append(ri)
        entry["lines"] += 1
    keys = sorted(chains, key=lambda key: (key[1], abs(key[0])))

    def column(f, dtype=float):
        return ("chain", np.array([f(k) for k in keys], dtype=dtype))

    out = xr.Dataset(
        {
            "n": column(lambda k: k[0], int),
            "m": column(lambda k: k[1], int),
            "iota": column(lambda k: k[0] / k[1]),
            "width": column(lambda k: chains[k]["width"]),
            "width_physical": column(lambda k: chains[k]["width_physical"]),
            "center": column(lambda k: np.mean(np.concatenate(chains[k]["radii"]))),
            "lines": column(lambda k: chains[k]["lines"], int),
            f"o_point_{pol}": column(lambda k: chains[k]["o_point"]),
        },
        coords={"chain": np.arange(len(keys))},
    )
    out["width"].attrs = {"long_name": f"island width in {radial}"}
    out["width_physical"].attrs = {"long_name": "island width in physical space"}
    out["center"].attrs = {"long_name": f"mean {radial} of the chain"}
    out.attrs = {**_provenance(section), "label": "island chains", "radial": radial}
    return out


# ---------------------------------------------------------------------------------------------
# Open field lines: connection lengths and footprints
# ---------------------------------------------------------------------------------------------
def footprint(lines: xr.Dataset) -> xr.Dataset:
    """Where the traced field lines left the grid, with their connection lengths.

    Parameters
    ----------
    lines : xarray.Dataset
        The lines of :func:`trace_field_lines`.

    Returns
    -------
    xarray.Dataset
        Over ``line``: the logical coordinates of the exit point (NaN for a line that did not
        leave the grid), ``x``, ``y``, ``z``, ``R``, ``connection_length`` and ``exited``, with
        the seeds as coordinates. The attrs name the dimensions.

    See Also
    --------
    plasma_plots.fieldline_plots.plot_footprint : The exit points over the angles.
    plasma_plots.fieldline_plots.plot_connection_length : The connection lengths over the seeds.

    Examples
    --------
    >>> hits = footprint(edge)
    >>> hits.where(hits.exited, drop=True).to_dataframe()
    """
    dims = _dims_of(lines)
    exited = np.asarray(lines["exited"], dtype=bool)
    data = {}
    for name in (*dims, "x", "y", "z"):
        values = np.asarray(lines[f"{name}_end"], dtype=float)
        attrs = dict(lines[name].attrs) if name in lines else {"label": f"${name}$"}
        data[name] = ("line", np.where(exited, values, np.nan), attrs)
    data["R"] = ("line", np.hypot(data["x"][1], data["y"][1]), {"label": "$R$"})
    data["connection_length"] = (
        "line",
        np.asarray(lines["connection_length"], dtype=float),
        dict(lines["connection_length"].attrs),
    )
    data["exited"] = ("line", exited)
    out = xr.Dataset(data, coords=_line_coords(lines))
    out.attrs = {
        **lines.attrs,
        "kind": "footprint",
        "label": f"footprint of {lines.attrs.get('label', 'field lines')}",
    }
    return out


def seed_grid(lines: xr.Dataset, name: str = "connection_length") -> xr.DataArray:
    """A per-line quantity over the two seed coordinates that vary, when the seeds form a grid.

    Parameters
    ----------
    lines : xarray.Dataset
        The lines of :func:`trace_field_lines` (or their :func:`footprint`), seeded with a dict of
        two 1-D arrays, so that the seeds form a grid.
    name : str, optional
        The per-line variable. Default: ``"connection_length"``.

    Returns
    -------
    xarray.DataArray
        ``name`` over the two varying seed coordinates (named after the logical dimensions, e.g.
        ``(theta, zeta)``). With ``direction="both"`` the forward lines are taken (both carry the
        same connection length).

    Raises
    ------
    ValueError
        If the seeds do not vary along exactly two coordinates, or do not fill a grid.

    Examples
    --------
    >>> seed_grid(edge, "connection_length").plasma.plot.slice()
    """
    dims = _dims_of(lines)
    if "direction" in lines.coords and (np.asarray(lines["direction"]) == -1).any():
        lines = lines.isel(line=np.flatnonzero(np.asarray(lines["direction"]) == 1))
    starts = {d: np.asarray(lines[f"{d}_start"], dtype=float) for d in dims}
    varying = [d for d in dims if np.unique(starts[d]).size > 1]
    if len(varying) != 2:
        raise ValueError(
            f"a seed grid needs seeds varying along exactly two coordinates; these vary along {varying}"
        )
    a, b = (np.unique(starts[d]) for d in varying)
    if a.size * b.size != lines.sizes["line"]:
        raise ValueError("the seeds do not fill a grid of the two varying coordinates")
    values = np.full((a.size, b.size), np.nan)
    ia = np.searchsorted(a, starts[varying[0]])
    ib = np.searchsorted(b, starts[varying[1]])
    values[ia, ib] = np.asarray(lines[name], dtype=float)
    out = xr.DataArray(
        values,
        dims=varying,
        coords={varying[0]: a, varying[1]: b},
        name=name,
        attrs={**_provenance(lines), **dict(lines[name].attrs)},
    )
    for d in varying:
        if d in lines:
            out[d].attrs = dict(lines[d].attrs)
    return out


# ---------------------------------------------------------------------------------------------
# A field along the lines
# ---------------------------------------------------------------------------------------------
def sample_along(field: xr.DataArray, lines: xr.Dataset) -> xr.DataArray:
    """Interpolate a scalar field along traced field lines, trilinearly on its logical grid.

    Parameters
    ----------
    field : xarray.DataArray
        A scalar field over the three logical dimensions the lines were traced in (its other
        dimensions, e.g. ``t``, are kept), with the ``X``, ``Y``, ``Z`` coordinates.
    lines : xarray.Dataset
        The lines of :func:`trace_field_lines`.

    Returns
    -------
    xarray.DataArray
        The field over ``(s, line)`` after the field's other dimensions, NaN where a line has
        stopped, named after the field and carrying its label and units; the ``s`` coordinate is
        the arc length, and the seeds, ``iota``, ``transits`` and ``connection_length`` are
        coordinates on ``line``.

    Raises
    ------
    ValueError
        If the field's logical dimensions differ from the lines', or it has a ``component``
        dimension.

    See Also
    --------
    parallel_wavenumber : The dominant wavenumber along each line.
    plasma_plots.fieldline_plots.plot_along_field_lines : The profiles.

    Examples
    --------
    >>> along = sample_along(phi, lines)  # (t, s, line)
    >>> along.isel(line=0).plasma.plot.slice(x="s", y="t")
    """
    validate_array(field)
    if "component" in field.dims:
        raise ValueError("sample_along takes a scalar field; select a component first")
    dims = _dims_of(lines)
    if tuple(logical_dims(field)) != dims or any(d not in field.dims for d in dims):
        raise ValueError(
            f"the field must be over the lines' logical dimensions {dims}; it has {field.dims}"
        )
    grid = _Grid(field)
    others = [d for d in field.dims if d not in dims]
    values = np.asarray(field.transpose(*others, *dims), dtype=float)
    eta = np.stack([np.asarray(lines[d], dtype=float) for d in dims], axis=-1)
    out = grid.values_at(values, eta)  # (*others, s, line)
    per_line = {
        name: ("line", np.asarray(lines[name]), dict(lines[name].attrs))
        for name in ("iota", "transits", "connection_length")
        if name in lines
    }
    return xr.DataArray(
        out,
        dims=(*others, "s", "line"),
        coords={
            **{d: field[d] for d in others if d in field.coords},
            "s": lines["s"],
            **_line_coords(lines),
            **per_line,
        },
        name=field.name,
        attrs={**field.attrs, "label": f"{_label(field)} along field lines".strip()},
    )


def parallel_wavenumber(
    samples: xr.DataArray, *, method: str = "fft", detrend: bool = True
) -> xr.DataArray:
    """The dominant wavenumber ``k∥`` of a field along each traced field line.

    From the samples of :func:`sample_along`, over the arc length ``s`` of each line (its valid,
    uniformly spaced part). ``"fft"`` takes the peak of the power spectrum (Hann window), refined
    by a parabola through the peak and its neighbours; ``"crossings"`` counts the zero crossings,
    ``k∥ = π × crossings / length``, which suits a few wavelengths. Compare with
    :func:`plasma_plots.theory.waves.parallel_wavenumber`, ``(n + m/q)/R₀`` for a mode
    ``(m, n)`` in a cylinder.

    Parameters
    ----------
    samples : xarray.DataArray
        The field along the lines, over ``(s, line)`` and any other dimensions.
    method : {"fft", "crossings"}, optional
        How to estimate it. Default: ``"fft"``.
    detrend : bool, optional
        Remove the mean along each line first. Default: True.

    Returns
    -------
    xarray.DataArray
        ``k_parallel`` over ``line`` and the other dimensions, NaN for a line with fewer than
        four samples.

    Raises
    ------
    ValueError
        If ``samples`` has no ``s`` dimension, or ``method`` is unknown.

    Examples
    --------
    >>> k_par = parallel_wavenumber(sample_along(phi.isel(t=-1), lines))
    """
    validate_array(samples, required_dims=("s",))
    if method not in ("fft", "crossings"):
        raise ValueError(f'method must be "fft" or "crossings"; got {method!r}')
    s = np.asarray(samples["s"], dtype=float)
    others = [d for d in samples.dims if d != "s"]
    values = np.asarray(samples.transpose(*others, "s"), dtype=float)
    flat = values.reshape(-1, s.size)
    out = np.full(flat.shape[0], np.nan)
    for i, row in enumerate(flat):
        valid = np.isfinite(row)
        n = int(valid.sum())
        if n < 4:
            continue
        y, x = row[valid], s[valid]
        if detrend:
            y = y - y.mean()
        if method == "crossings":
            crossings = np.count_nonzero(np.sign(y[1:]) * np.sign(y[:-1]) < 0)
            out[i] = np.pi * crossings / (x[-1] - x[0]) if x[-1] > x[0] else np.nan
            continue
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
        power = np.abs(np.fft.rfft(y * window)) ** 2
        k = 2 * np.pi * np.fft.rfftfreq(n, d=float(np.median(np.diff(x))))
        power[0] = 0.0
        j = int(np.argmax(power))
        if 0 < j < power.size - 1 and power[j] > 0:
            a, b, c = power[j - 1], power[j], power[j + 1]
            curvature = a - 2 * b + c
            shift = 0.5 * (a - c) / curvature if curvature != 0 else 0.0
            out[i] = k[j] + shift * (k[1] - k[0])
        else:
            out[i] = k[j]
    result = xr.DataArray(
        out.reshape(values.shape[:-1]),
        dims=others,
        coords={d: samples[d] for d in others if d in samples.coords},
        name="k_parallel",
        attrs={
            **_provenance(samples),
            "label": "$k_\\parallel$",
            "long_name": "parallel wavenumber",
        },
    )
    for name in samples.coords:
        if samples[name].dims == ("line",) and name not in result.coords:
            result = result.assign_coords({name: samples[name]})
    return result
