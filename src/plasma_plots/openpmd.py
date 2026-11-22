"""Read openPMD files into Datasets in plasma-plots' conventions.

`openPMD <https://www.openpmd.org>`_ is the file standard of WarpX, ImpactX, PIConGPU and other
particle-in-cell and tracking codes: particle species and meshes per iteration, every record
with its SI conversion (``unitSI``) and dimension (``unitDimension``). This module reads the HDF5
form with h5py (``pip install "plasma-plots[openpmd]"``), with no dependency on the code that
wrote it; ADIOS2 files (``.bp``) are converted first with openPMD-api's ``openpmd-pipe``
(``openpmd-pipe --infile run.bp --outfile run.h5``).

:func:`from_openpmd` returns the particles as a marker Dataset, the layout that
``markers.plasma.plot.phase_space`` and ``.scatter`` read:

* the dimension ``marker``, and ``t`` [s] with the ``iteration`` numbers beside it when more than
  one iteration is read;
* the position components with their offsets added (``x``, ``y``, ``z``, or the record's own
  component names, such as ImpactX's ``t``), the momentum as ``px``, ``py``, ``pz`` [kg m/s],
  and, where the species has a mass, the velocity ``vx``, ``vy``, ``vz`` [m/s]
  (relativistic: ``v = p / (m sqrt(1 + (p / m c)^2))``);
* every other record by its openPMD name (``weighting``, ``charge``, ``mass``, ``id``; a vector
  record ``r`` as ``r_x``, ...), constant records filled in;
* every value per real particle and in SI units, ``units`` from ``unitDimension``, and a
  ``label`` and ``long_name``;
* with several species, their markers one after the other and the variable ``species`` (the
  species' names), for ``phase_space(..., species="species")``.

Markers that carry an ``id`` are matched by it across iterations; without one, the ``k``-th
marker of every iteration is the same row. A marker missing from an iteration is NaN there.

:func:`openpmd_meshes` returns the mesh records of one iteration on their grids: the axes as
dimensions (named by ``axisLabels``) with coordinates in m, a vector record along
``component`` when its components share their staggering, and its components as separate
variables (``E_x``, ...) when they do not, each on coordinates offset by its ``position``
(``x_0.5`` for half a cell).

Examples
--------
>>> import plasma_plots
>>> beam = plasma_plots.from_openpmd(
...     "diags/openPMD/monitor.h5", species="beam"
... )
>>> beam.plasma.plot.phase_space(x="x", v="px", t=-1)
>>> both = plasma_plots.from_openpmd("diags/simData_%T.h5", iteration=-1)
>>> both.plasma.plot.phase_space(x="z", v="vz", species="species")
>>> fields = plasma_plots.openpmd_meshes("diags/simData_%T.h5", iteration=-1)
>>> fields.E.sel(component="z").plasma.plot.slice()
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import xarray as xr

#: The speed of light [m/s].
SPEED_OF_LIGHT = 299_792_458.0
#: The SI base units in the order of openPMD's ``unitDimension``.
BASE_UNITS = ("m", "kg", "s", "A", "K", "mol", "cd")
#: Names of common derived units, by their ``unitDimension``.
DERIVED_UNITS = {
    (1, 1, -3, -1, 0, 0, 0): "V/m",
    (2, 1, -3, -1, 0, 0, 0): "V",
    (0, 1, -2, -1, 0, 0, 0): "T",
    (0, 0, 1, 1, 0, 0, 0): "C",
    (-3, 0, 1, 1, 0, 0, 0): "C/m^3",
    (-2, 0, 0, 1, 0, 0, 0): "A/m^2",
    (2, 1, -2, 0, 0, 0, 0): "J",
    (1, 1, -1, 0, 0, 0, 0): "kg m/s",
    (1, 0, -1, 0, 0, 0, 0): "m/s",
    (-3, 0, 0, 0, 0, 0, 0): "m^-3",
}
#: How the particle records are named, labelled and described here.
_POSITION = ("{c}", "${c}$", "position {c}")
_MOMENTUM = ("p{c}", "$p_{{{c}}}$", "momentum {c}")
_VELOCITY = ("v{c}", "$v_{{{c}}}$", "velocity {c}")


def units_of(dimension: Sequence[float]) -> str:
    """The units of an openPMD ``unitDimension``, e.g. ``"kg m/s"`` or ``"m s^-2"``.

    Parameters
    ----------
    dimension : sequence of float
        The powers of length, mass, time, current, temperature, amount and luminous intensity.

    Returns
    -------
    str
        A common derived unit (``"V/m"``, ``"T"``, ``"C"``, ...) where one fits, else the base
        units with their powers; ``"1"`` for a dimensionless record.
    """
    powers = tuple(int(round(float(p))) for p in dimension)
    powers = (powers + (0,) * 7)[:7]
    if powers in DERIVED_UNITS:
        return DERIVED_UNITS[powers]
    if not any(powers):
        return "1"
    parts = [
        unit if power == 1 else f"{unit}^{power}"
        for unit, power in sorted(
            zip(BASE_UNITS, powers, strict=True), key=lambda item: item[1] < 0
        )
        if power
    ]
    return " ".join(parts)


def _text(value) -> str:
    """An HDF5 string attribute as ``str``."""
    if isinstance(value, bytes | np.bytes_):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        return _text(value[()])
    return str(value)


def _texts(value) -> list[str]:
    """An HDF5 attribute holding a list of strings (or one ``;``-separated string)."""
    if isinstance(value, bytes | np.bytes_ | str):
        return [part for part in _text(value).split(";") if part]
    return [_text(item) for item in np.atleast_1d(value)]


def _h5py():
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - h5py is in the dev extra
        raise ImportError(
            'reading openPMD files needs h5py: pip install "plasma-plots[openpmd]"'
        ) from error
    return h5py


def _pattern(text: str) -> tuple[re.Pattern, str]:
    """A regular expression for an iteration's file name pattern (``%T``, ``%06T``)."""
    match = re.search(r"%0?(\d*)T", text)
    if match is None:
        raise ValueError(f"{text!r} has no %T for the iteration number")
    return (
        re.compile(
            re.escape(text[: match.start()]) + r"(\d+)" + re.escape(text[match.end() :])
            + "$"
        ),
        text,
    )  # fmt: skip


class _Series:
    """The iterations of an openPMD series in HDF5: ``{iteration: (file, group path)}``."""

    @staticmethod
    def _matching(folder: Path, name: str) -> list[Path]:
        regex, _ = _pattern(name)
        files = sorted(f for f in folder.iterdir() if regex.match(f.name))
        if not files:
            raise FileNotFoundError(f"no files match {folder / name}")
        return files

    def _from_file(self, file: Path) -> list[Path]:
        h5py = _h5py()
        with h5py.File(file, "r") as h5:
            if "openPMD" not in h5.attrs:
                raise ValueError(
                    f"{file} is not an openPMD file (no openPMD attribute)"
                )
            encoding = _text(h5.attrs.get("iterationEncoding", "groupBased"))
            form = _text(h5.attrs.get("iterationFormat", ""))
        if encoding == "fileBased" and "%" in form:
            name = Path(form).name
            if not Path(name).suffix:  # openPMD-api writes the pattern without one
                name += file.suffix
            return self._matching(file.parent, name)
        if encoding == "variableBased":
            raise ValueError(
                f"{file}: variable-based iteration encoding is ADIOS2 only"
            )
        return [file]

    def __init__(self, path: str | Path) -> None:
        h5py = _h5py()
        text = str(path)
        if text.endswith(".bp") or text.endswith(".bp4") or text.endswith(".bp5"):
            raise ValueError(
                f"{text} is an ADIOS2 file; convert it to HDF5 first: "
                f"openpmd-pipe --infile {text} --outfile {Path(text).stem}.h5"
            )
        if "%" in Path(text).name:
            files = self._matching(Path(text).parent, Path(text).name)
        elif Path(text).is_dir():
            candidates = sorted(Path(text).glob("*.h5"))
            if not candidates:
                raise FileNotFoundError(f"no .h5 files in {text}")
            files = self._from_file(candidates[0])
        else:
            files = self._from_file(Path(text))
        #: The iterations, in increasing order, and where each one lives.
        self.iterations: dict[int, tuple[Path, str]] = {}
        for file in files:
            with h5py.File(file, "r") as h5:
                base = _text(h5.attrs.get("basePath", "/data/%T/"))
                self.meshes_path = _text(h5.attrs.get("meshesPath", "meshes/"))
                self.particles_path = _text(h5.attrs.get("particlesPath", "particles/"))
                root = base.split("%T")[0]
                for name in h5[root]:
                    if name.isdigit():
                        group = base.replace("%T", name)
                        self.iterations[int(name)] = (file, group)
        if not self.iterations:
            raise ValueError(f"{text} holds no openPMD iterations")
        self.iterations = dict(sorted(self.iterations.items()))

    def select(self, iteration) -> list[int]:
        """The iterations ``iteration`` names: a number, a negative index, a list, or all."""
        numbers = list(self.iterations)
        if iteration is None:
            return numbers
        if isinstance(iteration, int | np.integer):
            if iteration in self.iterations:
                return [int(iteration)]
            if iteration < 0:
                return [numbers[iteration]]
            raise KeyError(f"no iteration {iteration}; the series has {numbers}")
        return [n for item in iteration for n in self.select(int(item))]


def _component(item, count: int) -> np.ndarray:
    """A record component's values, unit-converted: a dataset, or a constant group."""
    unit = float(item.attrs.get("unitSI", 1.0))
    if "value" in item.attrs:  # a constant component
        shape = tuple(np.atleast_1d(item.attrs.get("shape", count)))
        return np.full(shape, float(np.asarray(item.attrs["value"])) * unit).reshape(-1)
    return np.asarray(item[()], dtype=float).reshape(-1) * unit


def _record(group, name: str, count: int) -> dict[str, np.ndarray]:
    """``{component: values}`` of a particle record (``""`` for a scalar record)."""
    item = group[name]
    h5py = _h5py()
    if isinstance(item, h5py.Dataset) or "value" in item.attrs:
        return {"": _component(item, count)}
    return {c: _component(item[c], count) for c in item}


def _count(group) -> int:
    """The number of particles of a species group."""
    h5py = _h5py()
    for name in group:
        item = group[name]
        if isinstance(item, h5py.Dataset):
            return int(item.size)
        if isinstance(item, h5py.Group):
            if "value" in item.attrs and "shape" in item.attrs:
                return int(np.prod(np.atleast_1d(item.attrs["shape"])))
            for component in item.values():
                if isinstance(component, h5py.Dataset):
                    return int(component.size)
                if "shape" in component.attrs:
                    return int(np.prod(np.atleast_1d(component.attrs["shape"])))
    return 0


def _species_variables(group) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """The variables and their attributes of one species at one iteration."""
    count = _count(group)
    records = {name: _record(group, name, count) for name in group}
    attributes = {name: dict(group[name].attrs) for name in group}
    weighting = records.get("weighting", {}).get("")

    def per_particle(name: str, values: np.ndarray) -> np.ndarray:
        if (
            name == "weighting"
        ):  # the number of real particles a macroparticle stands for
            return values
        attrs = attributes[name]
        power = float(np.asarray(attrs.get("weightingPower", 0.0)))
        if (
            int(np.asarray(attrs.get("macroWeighted", 0)))
            and power
            and weighting is not None
        ):
            return values / weighting**power
        return values

    out: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    offsets = records.get("positionOffset", {})
    for name, components in records.items():
        if name == "positionOffset":
            continue
        units = units_of(attributes[name].get("unitDimension", (0,) * 7))
        for component, values in components.items():
            values = per_particle(name, values)
            if name == "position":
                values = values + offsets.get(component, 0.0)
                key, label, long_name = (s.format(c=component) for s in _POSITION)
            elif name == "momentum":
                key, label, long_name = (s.format(c=component) for s in _MOMENTUM)
            elif component:
                key = f"{name}_{component}"
                label, long_name = key, f"{name} {component}"
            else:
                key, label, long_name = name, name, name
            out[key] = values
            meta[key] = {"units": units, "label": label, "long_name": long_name}
    if "momentum" in records and "mass" in out:
        mass = out["mass"]
        momenta = {c: out[_MOMENTUM[0].format(c=c)] for c in records["momentum"]}
        total = np.sqrt(sum(p**2 for p in momenta.values()))
        with np.errstate(divide="ignore", invalid="ignore"):
            gamma_m = mass * np.sqrt(1.0 + (total / (mass * SPEED_OF_LIGHT)) ** 2)
            for component, p in momenta.items():
                key, label, long_name = (s.format(c=component) for s in _VELOCITY)
                out[key] = p / gamma_m
                meta[key] = {"units": "m/s", "label": label, "long_name": long_name}
    return out, meta


def _iteration_time(group) -> float:
    """An iteration's time [s]."""
    time = float(np.asarray(group.attrs.get("time", 0.0)))
    return time * float(np.asarray(group.attrs.get("timeUnitSI", 1.0)))


def _concatenate(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """The species of one iteration one after the other; NaN where a species lacks a record."""
    sizes = [len(part["species"]) for part in parts]
    keys = sorted({key for part in parts for key in part})
    out = {}
    for key in keys:
        pieces = []
        for part, size in zip(parts, sizes, strict=True):
            if key in part:
                pieces.append(part[key])
            elif key == "species":  # pragma: no cover - every part has one
                pieces.append(np.full(size, "", dtype=object))
            else:
                pieces.append(np.full(size, np.nan))
        out[key] = np.concatenate(pieces) if pieces else np.zeros(0)
    return out


def _rows(frames: list[dict[str, np.ndarray]]) -> list[np.ndarray]:
    """Each iteration's row of every marker.

    A marker with an id is matched by ``(species, id)``, one without by its place among its
    species' markers in each iteration.
    """
    keys = []
    for frame in frames:
        species = frame.get("species", np.zeros(0, dtype=object))
        ids = frame.get("id", np.full(species.size, np.nan))
        place: dict[str, int] = {}
        frame_keys = []
        for name, number in zip(species, ids, strict=True):
            if np.isfinite(number):
                frame_keys.append((str(name), 0, int(number)))
            else:
                place[name] = place.get(name, -1) + 1
                frame_keys.append((str(name), 1, place[name]))
        keys.append(frame_keys)
    order = sorted(set().union(*keys)) if keys else []
    index = {key: n for n, key in enumerate(order)}
    return [np.array([index[k] for k in frame_keys], dtype=int) for frame_keys in keys]


def from_openpmd(
    path: str | Path,
    *,
    species: str | Sequence[str] | None = None,
    iteration: int | Sequence[int] | None = None,
) -> xr.Dataset:
    """Read the particles of an openPMD series as a marker Dataset.

    See :mod:`plasma_plots.openpmd` for what the Dataset contains.

    Parameters
    ----------
    path : str or pathlib.Path
        An HDF5 file of the series, the folder holding it, or a file pattern with ``%T`` for
        the iteration (``"diags/simData_%T.h5"``). A file of a file-based series finds the
        others by the series' ``iterationFormat``.
    species : str or sequence of str, optional
        The species to read. Default: all of them, with a ``species`` variable telling them
        apart.
    iteration : int or sequence of int, optional
        The iterations: their numbers, or negative indices counted from the last (``-1``). A
        single one gives no ``t`` dimension (``t`` and ``iteration`` are then scalar
        coordinates). Default: all.

    Returns
    -------
    xarray.Dataset
        The markers, over ``marker`` (and ``t``).

    Raises
    ------
    ValueError
        If the file is not openPMD, is ADIOS2, or holds none of the species.
    """
    h5py = _h5py()
    series = _Series(path)
    numbers = series.select(iteration)
    single = isinstance(iteration, int | np.integer)
    wanted = [species] if isinstance(species, str) else species
    frames: list[dict[str, np.ndarray]] = []
    meta: dict[str, dict] = {}
    times = []
    for number in numbers:
        file, group_path = series.iterations[number]
        with h5py.File(file, "r") as h5:
            group = h5[group_path]
            times.append(_iteration_time(group))
            particles = group.get(series.particles_path.rstrip("/"))
            names = [] if particles is None else sorted(particles)
            if wanted is not None:
                missing = [s for s in wanted if s not in names]
                if missing and len(numbers) == 1:
                    raise ValueError(
                        f"iteration {number} has no species {missing}; it has {names}"
                    )
                names = [s for s in wanted if s in names]
            parts = []
            for name in names:
                values, attrs = _species_variables(particles[name])
                meta.update(attrs)
                count = len(next(iter(values.values()), ()))
                values["species"] = np.full(count, name, dtype=object)
                parts.append(values)
            frames.append(_concatenate(parts))
    if not any(len(frame.get("species", ())) for frame in frames):
        raise ValueError(f"{path} holds no particles of species {species or 'any'}")
    keys = sorted({key for frame in frames for key in frame})
    rows = _rows(frames)
    size = max((int(r.max()) + 1 for r in rows if r.size), default=0)
    data = {}
    for key in keys:
        is_text = any(frame.get(key, np.zeros(0)).dtype == object for frame in frames)
        table = np.full(
            (len(frames), size),
            None if is_text else np.nan,
            dtype=object if is_text else float,
        )
        for k, (frame, row) in enumerate(zip(frames, rows, strict=True)):
            if key in frame:
                table[k, row] = frame[key]
        if is_text:
            # a marker's species is that of the iteration where it appears first
            first = table[0].copy()
            for k in range(1, len(frames)):
                empty = np.array([v is None for v in first])
                first[empty] = table[k][empty]
            data[key] = (("marker",), first.astype(str))
        else:
            data[key] = (("t", "marker"), table, meta.get(key, {}))
    dataset = xr.Dataset(
        data,
        coords={
            "t": (
                "t",
                np.array(times),
                {"units": "s", "label": "$t$", "long_name": "time"},
            ),
            "iteration": ("t", np.array(numbers)),
        },
    )
    if "id" in dataset:
        dataset["id"] = dataset["id"].max("t")
        dataset = dataset.set_coords("id")
    if single:
        dataset = dataset.isel(t=0)
    return dataset


def _mesh(item, name: str) -> dict[str, xr.DataArray]:
    """The variables of one mesh record: one along ``component``, or one per component."""
    h5py = _h5py()
    attrs = dict(item.attrs)
    if isinstance(item, h5py.Dataset) or "value" in item.attrs:
        components = {"": item}
    else:
        components = {c: item[c] for c in item}
    labels = _texts(attrs.get("axisLabels", ()))
    order = _text(attrs.get("dataOrder", "C"))
    geometry = _text(attrs.get("geometry", "cartesian"))
    spacing = np.atleast_1d(np.asarray(attrs.get("gridSpacing", 1.0), dtype=float))
    offset = np.atleast_1d(np.asarray(attrs.get("gridGlobalOffset", 0.0), dtype=float))
    unit = float(np.asarray(attrs.get("gridUnitSI", 1.0)))
    units = units_of(attrs.get("unitDimension", (0,) * 7))
    if order == "F":  # the labels and grid attributes are listed fastest axis first
        labels, spacing, offset = labels[::-1], spacing[::-1], offset[::-1]
    arrays = {}
    for component, data in components.items():
        position = np.atleast_1d(
            np.asarray(data.attrs.get("position", np.zeros(len(labels))), dtype=float)
        )
        if order == "F":
            position = position[::-1]
        if "value" in data.attrs:
            shape = tuple(int(n) for n in np.atleast_1d(data.attrs["shape"]))
            values = np.full(shape, float(np.asarray(data.attrs["value"])))
        else:
            values = np.asarray(data[()], dtype=float)
        values = values * float(np.asarray(data.attrs.get("unitSI", 1.0)))
        spatial = values.shape[-len(labels) :] if labels else values.shape
        dims, coords = [], {}
        if geometry == "thetaMode" and values.ndim == len(labels) + 1:
            dims.append("mode")
        for axis, label in enumerate(labels):
            stagger = float(position[axis]) if axis < position.size else 0.0
            dim = label if stagger == 0.0 else f"{label}_{stagger:g}"
            dims.append(dim)
            coords[dim] = (
                dim,
                (offset[axis] + (np.arange(spatial[axis]) + stagger) * spacing[axis])
                * unit,
                {"units": "m", "label": f"${label}$"},
            )
        arrays[component] = xr.DataArray(
            values,
            dims=dims,
            coords=coords,
            attrs={
                "units": units,
                "label": name if not component else f"${name}_{{{component}}}$",
                "long_name": name if not component else f"{name} {component}",
                "geometry": geometry,
                "position": position,
            },
        )
    if list(arrays) == [""]:
        array = arrays[""]
        array.attrs.update(label=name)
        return {name: array}
    shared = all(a.dims == next(iter(arrays.values())).dims for a in arrays.values())
    if shared:
        stacked = xr.concat(
            list(arrays.values()),
            dim=xr.DataArray(list(arrays), dims="component", name="component"),
        )
        stacked.attrs.update(label=name, long_name=name)
        stacked.attrs.pop("position", None)
        return {name: stacked}
    return {f"{name}_{component}": array for component, array in arrays.items()}


def openpmd_meshes(
    path: str | Path,
    *,
    names: str | Sequence[str] | None = None,
    iteration: int = -1,
) -> xr.Dataset:
    """Read the mesh records of one iteration of an openPMD series.

    See :mod:`plasma_plots.openpmd` for the layout.

    Parameters
    ----------
    path : str or pathlib.Path
        As for :func:`from_openpmd`.
    names : str or sequence of str, optional
        The records (``"E"``, ``"rho"``, ...). Default: all of them.
    iteration : int, optional
        The iteration number, or a negative index from the last. Default: the last.

    Returns
    -------
    xarray.Dataset
        The records on their grids, with ``t`` and ``iteration`` as scalar coordinates. In
        ``thetaMode`` geometry the first dimension is ``mode``, the azimuthal modes' real and
        imaginary parts as openPMD stores them.

    Raises
    ------
    ValueError
        If the iteration has no meshes, or none of ``names``.
    """
    h5py = _h5py()
    series = _Series(path)
    (number,) = series.select(int(iteration))
    file, group_path = series.iterations[number]
    wanted = [names] if isinstance(names, str) else names
    variables: dict[str, xr.DataArray] = {}
    with h5py.File(file, "r") as h5:
        group = h5[group_path]
        time = _iteration_time(group)
        meshes = group.get(series.meshes_path.rstrip("/"))
        available = [] if meshes is None else sorted(meshes)
        chosen = available if wanted is None else [n for n in wanted if n in available]
        if not chosen:
            raise ValueError(
                f"iteration {number} has no meshes {wanted or ''}; it has {available}"
            )
        for name in chosen:
            variables.update(_mesh(meshes[name], name))
    dataset = xr.Dataset(variables)
    return dataset.assign_coords(
        t=((), time, {"units": "s", "label": "$t$", "long_name": "time"}),
        iteration=number,
    )
