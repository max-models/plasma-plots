"""Generate the figures of the "Stellarator coils" guide (docs/src/assets/figures/coils_*).

The guide is about coilsets in the layout of the CoilStellaration dataset and plasma boundaries
in the layout of ConStellaration. So that the docs build needs no download, the coilset here
is made by DESC itself around its W7-X example (``desc.coils.initialize_modular_coils``,
converted to ``FourierXYZCoil``): planar coils, the kind of start a coil optimization begins
from, not an optimized coilset. It goes through the same functions as a coilset from the
dataset: written out in the dataset's layout, read back with ``plasma_plots.from_coilset``; the
boundary goes through ``boundary_from_desc`` and ``plasma_plots.from_boundary``. The field
error on the plasma (the benchmark's first metric) comes from DESC's Biot-Savart field of the
coils. It needs ``pip install desc-opt`` and PyVista (see .github/workflows/docs.yml).

Run from the repo root: python scripts/generate_coil_figures.py (or: make figures).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import desc.examples
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from desc.coils import initialize_modular_coils
from desc.grid import LinearGrid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plasma_plots  # noqa: E402
from plasma_plots.coils import (  # noqa: E402
    boundary_from_desc,
    coil_to_coil_distance,
    distance_to_surface,
)
from plasma_plots.plotting import save_figure  # noqa: E402

warnings.filterwarnings("ignore", module="desc")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "src" / "assets" / "figures"
NUMBERS = DOCS / "src" / "assets" / "coils" / "numbers.txt"
PYVISTA_OUT = DOCS / "public" / "pyvista"
for folder in (OUT, NUMBERS.parent, PYVISTA_OUT):
    folder.mkdir(parents=True, exist_ok=True)


def save(result, filename):
    """Save a figure with a dark variant ``<stem>.dark.<ext>`` for the theme's dark mode."""
    stem, ext = str(filename).rsplit(".", 1)
    for path in save_figure(result, OUT / stem, formats=(ext,), dark=True):
        print(f"wrote {path}")
    plt.close(result.fig)


# =============================================================================
# A coilset and a boundary in the datasets' layouts
# =============================================================================
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    w7x = desc.examples.get("W7-X")
desc_coils = initialize_modular_coils(
    w7x, num_coils=5, r_over_a=2.5
).to_FourierXYZ(N=3)

# a row of CoilStellaration's "coilsets" subset holds this dict as JSON
coilset = {
    "coil_x_n": [np.asarray(coil.X_n).tolist() for coil in desc_coils.coils],
    "coil_y_n": [np.asarray(coil.Y_n).tolist() for coil in desc_coils.coils],
    "coil_z_n": [np.asarray(coil.Z_n).tolist() for coil in desc_coils.coils],
    "currents": [float(coil.current) for coil in desc_coils.coils],
    "n_field_periods": int(desc_coils.NFP),
    "is_stellarator_symmetric": bool(desc_coils.sym),
}
# and ConStellaration's "boundary.json" this one
boundary = boundary_from_desc(w7x)

coils = plasma_plots.from_coilset(json.dumps(coilset), n_points=128)
plasma = plasma_plots.from_boundary(json.dumps(boundary), theta=96)
a = plasma.attrs["minor_radius"]
print(coils)

# =============================================================================
# The benchmark's metrics
# =============================================================================
coils["plasma_distance"] = distance_to_surface(coils, plasma) / a
coils["plasma_distance"].attrs.update(label="$d_p / a$", units="1")
coils["coil_distance"] = coil_to_coil_distance(coils) / a
coils["coil_distance"].attrs.update(label="$d_c / a$", units="1")

# the field error |B.n| / |B| on the plasma, with DESC's field of the coils
surface = plasma_plots.from_desc(
    w7x,
    "n_rho",
    rho=[1.0],
    theta=64,
    zeta=np.linspace(0, 2 * np.pi, 48 * w7x.NFP, endpoint=False),
)
points = np.stack([surface[axis].values.ravel() for axis in "XYZ"], axis=-1)
B = np.asarray(
    desc_coils.compute_magnetic_field(
        points, basis="xyz", source_grid=LinearGrid(N=64)
    )
)
n = surface.n_rho  # the unit normal, Cartesian components along "component"
B = xr.DataArray(B.T.reshape(n.shape), dims=n.dims, coords=n.coords)
field_error = abs((B * n).sum("component")) / np.sqrt((B**2).sum("component"))
field_error.attrs = {"label": r"$|B \cdot n| / |B|$", "units": "1"}

rows = [
    ("minor radius a from from_boundary (m)", a),
    ("minor radius a, DESC's own (m)", float(w7x.compute("a")["a"])),
    (
        "coils (unique, all)",
        f"{len(coilset['currents'])}, {coils.sizes['coil']}",
    ),
    ("max curvature * a", float(coils.curvature.max()) * a),
    ("min coil-plasma distance / a", float(coils.plasma_distance.min())),
    ("min coil-coil distance / a", float(coils.coil_distance.min())),
    ("mean field error |B.n|/|B|", float(field_error.mean())),
]
NUMBERS.write_text(
    "".join(
        f"{label:<40} {value:.6f}\n"
        if isinstance(value, float)
        else f"{label:<40} {value}\n"
        for label, value in rows
    )
)
print(NUMBERS.read_text())

# =============================================================================
# Metrics along the coils (Matplotlib)
# =============================================================================
unique = coils.isel(coil=slice(0, len(coilset["currents"])))
save(
    unique.plasma_distance.plasma.plot.slice(x="s", y="coil"),
    "coils_plasma_distance_map.png",
)

# =============================================================================
# Interactive 3-D: the coils around the plasma
# =============================================================================
try:
    import pyvista as pv

    pv.OFF_SCREEN = True

    def shot(plotter, filename, *, zoom=1.0):
        plotter.camera_position = "iso"
        plotter.camera.zoom(zoom)
        plotter.screenshot(str(OUT / filename), window_size=[1000, 700])
        print(f"wrote {OUT / filename}")
        html = PYVISTA_OUT / filename.replace(".png", ".html")
        # vtk.js would show them 0-255
        for title in list(plotter.scalar_bars.keys()):
            plotter.remove_scalar_bar(title)
        try:
            plotter.trame.export_html(str(html))
            print(f"wrote {html}")
        except Exception as exc:  # needs trame-pyvista
            print(
                f"skipped {html.name} (interactive export unavailable): {exc}"
            )
        plotter.close()

    shot(
        coils.plasma.plot.coils_3d(surface=field_error, surface_cmap="magma"),
        "coils_3d_field_error.png",
        zoom=1.3,
    )
    shot(
        coils.plasma.plot.coils_3d(color_by="plasma_distance", surface=plasma),
        "coils_3d_distance.png",
        zoom=1.3,
    )
except ImportError as exc:  # pragma: no cover - optional
    print(f"skipped the 3-D views (pyvista unavailable): {exc}")
