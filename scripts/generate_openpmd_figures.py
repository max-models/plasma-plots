"""Generate the figures of the "openPMD files" guide (docs/public/plotly/plotly_openpmd_*).

It writes a small openPMD series by hand with h5py, as a PIC or tracking code would: an H-
beamlet and its co-extracted electrons leaving an extraction gap, three iterations of a
focusing drift, and the charge density on a mesh. It then reads the series back with
``plasma_plots.from_openpmd`` and ``plasma_plots.openpmd_meshes`` and plots it. Only h5py is
needed (``pip install "plasma-plots[openpmd]"``); a file written by WarpX, ImpactX or
openPMD-api reads the same way.

Run from the repo root: python scripts/generate_openpmd_figures.py (or: make figures).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plasma_plots  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PLOTLY_OUT = ROOT / "docs" / "public" / "plotly"
PLOTLY_OUT.mkdir(parents=True, exist_ok=True)

E = 1.602176634e-19
C = 299_792_458.0
MASS = {"H-": 1.00837 * 1.66053906660e-27, "electrons": 9.1093837015e-31}
ENERGY = 45e3  # eV: the H- beam of an ion source
ITERATIONS = (0, 500, 1000)
DT = 1e-11  # s per step


def _attrs(item, values: dict) -> None:
    for key, value in values.items():
        item.attrs[key] = np.bytes_(value) if isinstance(value, str) else value


def _record(item, dimension, weighting_power=0.0, macro=0) -> None:
    _attrs(
        item,
        {
            "unitDimension": np.array(dimension, dtype=float),
            "timeOffset": np.float32(0.0),
            "weightingPower": float(weighting_power),
            "macroWeighted": np.uint32(macro),
        },
    )


def _beam(name: str, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Positions and momenta at the start: a round beam, converging, with a thermal spread."""
    mass = MASS[name]
    speed = np.sqrt(2 * ENERGY * E / mass)
    momentum = (
        mass * speed * np.sqrt(1 + 0.0)
    )  # non-relativistic for H-, enough for this
    radius = 4e-3 if name == "H-" else 5e-3
    xy = rng.normal(scale=radius / 2, size=(2, n))
    angle = -0.02 * xy / radius + rng.normal(scale=0.004, size=(2, n))
    position = np.vstack([xy, np.zeros(n)])
    p = momentum * np.vstack([angle, np.ones(n)])
    return position, p


def write_series(path: Path) -> None:
    """A group-based openPMD file: two species over three iterations and a charge density."""
    rng = np.random.default_rng(4)
    start = {
        name: _beam(name, n, rng)
        for name, n in (("H-", 3000), ("electrons", 600))
    }
    with h5py.File(path, "w") as h5:
        _attrs(
            h5,
            {
                "openPMD": "1.1.0", "openPMDextension": np.uint32(0),
                "basePath": "/data/%T/", "meshesPath": "meshes/",
                "particlesPath": "particles/", "iterationEncoding": "groupBased",
                "iterationFormat": "/data/%T/", "software": "plasma-plots docs",
            },
        )  # fmt: skip
        for iteration in ITERATIONS:
            group = h5.create_group(f"data/{iteration}")
            _attrs(
                group, {"time": float(iteration), "dt": 1.0, "timeUnitSI": DT}
            )
            for name, (position, momentum) in start.items():
                mass = MASS[name]
                gamma = np.sqrt(
                    1 + (np.linalg.norm(momentum, axis=0) / (mass * C)) ** 2
                )
                x = position + momentum / (gamma * mass) * iteration * DT
                # the electrons leave the beam: those past 5 cm are dumped
                keep = (
                    np.flatnonzero(x[2] < 0.05)
                    if name == "electrons"
                    else slice(None)
                )
                species = group.create_group(f"particles/{name}")
                for record, values, dimension in (
                    ("position", x, (1, 0, 0, 0, 0, 0, 0)),
                    ("momentum", momentum, (1, 1, -1, 0, 0, 0, 0)),
                ):
                    item = species.create_group(record)
                    _record(item, dimension)
                    for axis, component in enumerate("xyz"):
                        data = item.create_dataset(
                            component, data=values[axis][keep]
                        )
                        data.attrs["unitSI"] = 1.0
                count = x[0][keep].size
                offset = species.create_group("positionOffset")
                _record(offset, (1, 0, 0, 0, 0, 0, 0))
                for component in "xyz":
                    constant = offset.create_group(component)
                    # fmt: skip
                    _attrs(
                        constant,
                        {
                            "value": 0.0,
                            "shape": np.array([count], dtype=np.uint64),
                            "unitSI": 1.0,
                        },
                    )
                for record, value, dimension in (
                    ("charge", -E, (0, 0, 1, 1, 0, 0, 0)),
                    ("mass", mass, (0, 1, 0, 0, 0, 0, 0)),
                ):
                    constant = species.create_group(record)
                    _record(constant, dimension)
                    # fmt: skip
                    _attrs(
                        constant,
                        {
                            "value": value,
                            "shape": np.array([count], dtype=np.uint64),
                            "unitSI": 1.0,
                        },
                    )
                weighting = species.create_dataset(
                    "weighting", data=np.full(count, 2e5)
                )
                _record(weighting, (0, 0, 0, 0, 0, 0, 0), 1.0, 1)
                weighting.attrs["unitSI"] = 1.0
            # the H- charge density on an (x, z) mesh through the axis
            h = 1e-3
            xs, zs = np.arange(-10, 11) * h, np.arange(0, 121) * h
            position, momentum = start["H-"]
            velocity = momentum / MASS["H-"]
            rho = np.zeros((xs.size, zs.size))
            for k in range(zs.size):
                along = position + velocity * (zs[k] / velocity[2])
                near = np.abs(along[1]) < 2 * h
                counts, _ = np.histogram(
                    along[0][near],
                    bins=xs.size,
                    range=(xs[0] - h / 2, xs[-1] + h / 2),
                )
                rho[:, k] = -counts
            mesh = group.create_dataset("meshes/rho", data=rho / rho.min())
            _record(mesh, (-3, 0, 1, 1, 0, 0, 0))
            _attrs(
                mesh,
                {
                    "geometry": "cartesian", "dataOrder": "C",
                    "axisLabels": np.array([b"x", b"z"]), "gridSpacing": np.array([h, h]),
                    "gridGlobalOffset": np.array([xs[0], 0.0]), "gridUnitSI": 1.0,
                    "position": np.array([0.0, 0.0]), "unitSI": -1e-6,
                },
            )  # fmt: skip


def save_plotly(result, filename):
    """A web figure and its posters for light and dark mode (kaleido)."""
    print(f"wrote {result.save(PLOTLY_OUT / f'{filename}.plotly.json')}")


def main() -> None:
    plasma_plots.set_backend("plotly")
    plasma_plots.set_math("latex")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "beamlet.h5"
        write_series(path)
        markers = plasma_plots.from_openpmd(path)
        print(markers)
        save_plotly(
            markers.plasma.plot.phase_space(
                x="x", v="px", species="species", t=-1
            ),
            "plotly_openpmd_phase_space",
        )
        save_plotly(
            markers.sel(
                marker=markers.species == "H-"
            ).plasma.plot.phase_space(
                x="x",
                v="vx",
                kind="histogram",
                weight="weighting",
                bins=48,
                t=-1,
            ),
            "plotly_openpmd_histogram",
        )
        fields = plasma_plots.openpmd_meshes(path, iteration=0)
        save_plotly(fields.rho.plasma.plot.slice(x="z"), "plotly_openpmd_rho")


if __name__ == "__main__":
    main()
