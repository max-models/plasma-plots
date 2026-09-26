"""Generate the example figures embedded in docs/src/assets/figures/.

Builds small, purely synthetic labeled xarray data (no struphy runtime needed)
and renders it with struphy_plots, so the docs show real output of the actual
plotting code rather than mockups.

Run from the repo root: python scripts/generate_docs_figures.py
(or: make figures)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import struphy_plots  # noqa: F401  (registers .struphy on DataArray/Dataset)

OUT = Path(__file__).resolve().parents[1] / "docs" / "src" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def field_array(name, label, unit, values, dims, coords, coord_units=None):
    array = xr.DataArray(values, dims=dims, coords=coords, name=name, attrs={"label": label, "units": unit})
    for dim, unit_ in (coord_units or {}).items():
        if dim in array.coords and unit_:
            array.coords[dim].attrs["units"] = unit_
    return array


def save(result, filename):
    path = OUT / filename
    result.save(path, close=True)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# A wave packet orbiting the unit square, used for slice/panels/vector figures
# ---------------------------------------------------------------------------
n_t, n_e1, n_e2 = 24, 96, 96
t = np.linspace(0.0, 2 * np.pi, n_t)
e1 = np.linspace(0.0, 1.0, n_e1)
e2 = np.linspace(0.0, 1.0, n_e2)
E1, E2 = np.meshgrid(e1, e2, indexing="ij")

radius = 0.28
center1 = 0.5 + radius * np.cos(t)
center2 = 0.5 + radius * np.sin(t)
sigma = 0.10

phi = np.empty((n_t, n_e1, n_e2))
for i in range(n_t):
    dist2 = (E1 - center1[i]) ** 2 + (E2 - center2[i]) ** 2
    phi[i] = np.exp(-dist2 / (2 * sigma**2)) * np.cos(10.0 * (E1 - center1[i]))

field = field_array(
    "phi",
    r"$\phi$",
    "a.u.",
    phi,
    ("t", "e1", "e2"),
    {"t": t, "e1": e1, "e2": e2},
)

save(field.struphy.plot.slice(x="e1", y="e2", t="last"), "slice.png")
save(field.struphy.plot.panels(x="e1", y="e2", nrows=2, ncols=3), "panels.png")
save(field.struphy.plot.lineout(x="e1", t="last", e2=0.5), "lineout.png")

# ---------------------------------------------------------------------------
# A solid-body rotation vector field on the same grid
# ---------------------------------------------------------------------------
vx = -(E2 - 0.5)
vy = E1 - 0.5
vector = field_array(
    "v",
    r"$\mathbf{v}$",
    "a.u.",
    np.stack([np.broadcast_to(vx, (n_t, n_e1, n_e2)), np.broadcast_to(vy, (n_t, n_e1, n_e2))], axis=1),
    ("t", "component", "e1", "e2"),
    {"t": t, "e1": e1, "e2": e2},
)
save(vector.struphy.plot.vector(x="e1", y="e2", components=(0, 1), stride=6, t="last"), "vector.png")

# ---------------------------------------------------------------------------
# Two energy time series: a growth phase followed by saturation
# ---------------------------------------------------------------------------
tt = np.linspace(0.0, 6.0, 200)
rate = 0.6
en_phi = np.exp(rate * tt) / (1.0 + 0.02 * np.exp(rate * tt)) + 1e-3 * np.cos(20 * tt)
en_tot = 1.0 + 0.02 * (1.0 - np.exp(-0.5 * tt)) + 2e-3 * np.sin(15 * tt)

energy = field_array("en_phi", r"$e_\phi$", "J", en_phi, ("t",), {"t": tt})
total = field_array("en_tot", r"$e_{tot}$", "J", en_tot, ("t",), {"t": tt})

save(energy.struphy.plot.timeseries(fit=(0.0, 2.0), title="Field energy growth"), "timeseries.png")
save(
    struphy_plots.plotting.plot_scalars({"en_phi": energy, "en_tot": total}, logy=False),
    "scalars.png",
)

# ---------------------------------------------------------------------------
# Two "runs" of the same diagnostic, compared
# ---------------------------------------------------------------------------
run_a = field_array("en_phi", r"$e_\phi$", "J", np.exp(0.55 * tt), ("t",), {"t": tt})
run_b = field_array("en_phi", r"$e_\phi$", "J", np.exp(0.62 * tt), ("t",), {"t": tt})
save(run_a.struphy.plot.compare(run_b, mode="ratio"), "compare.png")

# ---------------------------------------------------------------------------
# Helical marker trajectories (the new per-quantity orbits Dataset shape)
# ---------------------------------------------------------------------------
n_steps, n_markers = 80, 24
s = np.linspace(0.0, 4 * np.pi, n_steps)
rng = np.random.default_rng(0)
radii = 0.2 + 0.15 * rng.random(n_markers)
phases = rng.uniform(0, 2 * np.pi, n_markers)
pitch = 0.15 + 0.1 * rng.random(n_markers)

X = radii[None, :] * np.cos(s[:, None] + phases[None, :])
Y = radii[None, :] * np.sin(s[:, None] + phases[None, :])
Z = pitch[None, :] * s[:, None] / (2 * np.pi)

orbits = xr.Dataset(
    {
        "x": (("t", "marker"), X),
        "y": (("t", "marker"), Y),
        "z": (("t", "marker"), Z),
    },
    coords={"t": s, "marker": np.arange(n_markers)},
    attrs={"product": "orbits", "label": "marker orbits"},
)
save(orbits.struphy.plot.trajectories(show_paths=True), "trajectories.png")

print("done")
