"""Generate the figures of the "Stellarator coils" guide (docs/src/assets/figures/coils_*).

The guide is about coilsets in the layout of the CoilStellaration dataset and plasma boundaries
in the layout of ConStellaration. So that the docs build needs no download, the coilsets here
are made by DESC itself around its W7-X example: circular planar coils
(``desc.coils.initialize_modular_coils``, converted to ``FourierXYZCoil``), the start of a coil
optimization, and what a short DESC optimization makes of them (fifteen iterations against the
field error, with bounds on the curvature, the length and the distances, the kinds of
CoilStellaration's DESC stage; about a minute and a half on four cores). Both go through the
same functions as a coilset from the dataset: written out in the dataset's layout, read back
with ``plasma_plots.from_coilset``; the boundary goes through ``boundary_from_desc`` and
``plasma_plots.from_boundary``. The field error on the plasma (the benchmark's first metric)
comes from DESC's Biot-Savart field of the coils. It needs ``pip install desc-opt`` and PyVista
(see .github/workflows/docs.yml).

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
from desc.objectives import (
    CoilCurvature,
    CoilLength,
    CoilSetMinDistance,
    FixCoilCurrent,
    ObjectiveFunction,
    PlasmaCoilSetMinDistance,
    QuadraticFlux,
)
from desc.optimize import Optimizer

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
# Coilsets and a boundary in the datasets' layouts
# =============================================================================
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    w7x = desc.examples.get("W7-X")
a_desc = float(w7x.compute("a")["a"])

# circles around the magnetic axis, five per half period; N=4 leaves the optimizer room
start_coils = initialize_modular_coils(
    w7x, num_coils=5, r_over_a=2.5
).to_FourierXYZ(N=4)


def optimize_coils(coils, maxiter=15):
    """A short DESC coil optimization: the field error, with the benchmark's kinds of bounds."""
    coil_grid = LinearGrid(N=24)
    plasma_grid = LinearGrid(M=12, N=12, NFP=w7x.NFP, sym=w7x.sym)
    length = float(np.mean(coils.compute("length")[0]["length"]))
    absolute = {"normalize": False, "normalize_target": False}
    objective = ObjectiveFunction(
        (
            QuadraticFlux(
                w7x,
                coils,
                eval_grid=plasma_grid,
                field_grid=coil_grid,
                vacuum=True,
                weight=50,
            ),
            # bends no sharper than the minor radius, so the coils stay smooth
            CoilCurvature(
                coils,
                bounds=(-np.inf, 1 / a_desc),
                grid=coil_grid,
                weight=100,
                **absolute,
            ),
            CoilLength(
                coils,
                bounds=(0, 1.1 * length),
                grid=coil_grid,
                weight=10,
                **absolute,
            ),
            CoilSetMinDistance(
                coils,
                bounds=(0.4 * a_desc, np.inf),
                grid=coil_grid,
                weight=10,
                **absolute,
            ),
            PlasmaCoilSetMinDistance(
                w7x,
                coils,
                bounds=(a_desc, np.inf),
                plasma_grid=plasma_grid,
                coil_grid=coil_grid,
                eq_fixed=True,
                weight=10,
                **absolute,
            ),
        ),
        deriv_mode="batched",
        jac_chunk_size=32,  # the full Jacobian of the Biot-Savart field needs ~20 GB
    )
    (optimized,), _ = Optimizer("lsq-exact").optimize(
        coils,
        objective=objective,
        constraints=(FixCoilCurrent(coils),),
        maxiter=maxiter,
        ftol=1e-6,
        xtol=1e-8,
        verbose=1,
        copy=True,
    )
    return optimized


def as_coilset(desc_coils):
    """A row of CoilStellaration's "coilsets" subset holds this dict as JSON."""
    return {
        "coil_x_n": [np.asarray(c.X_n).tolist() for c in desc_coils.coils],
        "coil_y_n": [np.asarray(c.Y_n).tolist() for c in desc_coils.coils],
        "coil_z_n": [np.asarray(c.Z_n).tolist() for c in desc_coils.coils],
        "currents": [float(c.current) for c in desc_coils.coils],
        "n_field_periods": int(desc_coils.NFP),
        "is_stellarator_symmetric": bool(desc_coils.sym),
    }


optimized_coils = optimize_coils(start_coils)
coilsets = {
    "start": as_coilset(start_coils),
    "optimized": as_coilset(optimized_coils),
}
# and ConStellaration's "boundary.json" this one
boundary = boundary_from_desc(w7x)
plasma = plasma_plots.from_boundary(json.dumps(boundary), theta=96)
a = plasma.attrs["minor_radius"]

# =============================================================================
# The benchmark's metrics
# =============================================================================
# the field error |B.n| / |B| on the plasma, with DESC's field of the coils
surface = plasma_plots.from_desc(
    w7x,
    "n_rho",
    rho=[1.0],
    theta=64,
    zeta=np.linspace(0, 2 * np.pi, 48 * w7x.NFP, endpoint=False),
)
points = np.stack([surface[axis].values.ravel() for axis in "XYZ"], axis=-1)
n = surface.n_rho  # the unit normal, Cartesian components along "component"


def field_error_of(desc_coils):
    B = np.asarray(
        desc_coils.compute_magnetic_field(
            points, basis="xyz", source_grid=LinearGrid(N=64)
        )
    )
    B = xr.DataArray(B.T.reshape(n.shape), dims=n.dims, coords=n.coords)
    error = abs((B * n).sum("component")) / np.sqrt((B**2).sum("component"))
    error.attrs = {"label": r"$|B \cdot n| / |B|$", "units": "1"}
    return error


coils, field_error = {}, {}
for name, desc_coils in (
    ("start", start_coils),
    ("optimized", optimized_coils),
):
    ds = plasma_plots.from_coilset(json.dumps(coilsets[name]), n_points=128)
    ds["plasma_distance"] = distance_to_surface(ds, plasma) / a
    ds["plasma_distance"].attrs.update(label="$d_p / a$", units="1")
    ds["coil_distance"] = coil_to_coil_distance(ds) / a
    ds["coil_distance"].attrs.update(label="$d_c / a$", units="1")
    coils[name] = ds
    field_error[name] = field_error_of(desc_coils)
print(coils["optimized"])

rows = [
    ("minor radius a from from_boundary (m)", a),
    ("minor radius a, DESC's own (m)", a_desc),
    (
        "coils (unique, all)",
        f"{len(coilsets['start']['currents'])}, {coils['start'].sizes['coil']}",
    ),
    ("", "start      optimized"),
]
for label, metric in (
    ("max curvature * a", lambda c, e: float(c.curvature.max()) * a),
    (
        "min coil-plasma distance / a",
        lambda c, e: float(c.plasma_distance.min()),
    ),
    ("min coil-coil distance / a", lambda c, e: float(c.coil_distance.min())),
    ("mean field error |B.n|/|B|", lambda c, e: float(e.mean())),
    ("max field error |B.n|/|B|", lambda c, e: float(e.max())),
):
    values = [metric(coils[k], field_error[k]) for k in ("start", "optimized")]
    rows.append((label, "   ".join(f"{v:.6f}" for v in values)))
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
unique = coils["optimized"].isel(
    coil=slice(0, len(coilsets["optimized"]["currents"]))
)
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

    # one logarithmic color scale for both, so the two field errors compare
    log_error = {
        name: np.log10(error).assign_attrs(
            label=r"$\log_{10} |B \cdot n| / |B|$", units="1"
        )
        for name, error in field_error.items()
    }
    clim = (-4.0, 0.0)  # 1e-4 to 1; the error passes through zero along lines
    for name, filename in (
        ("start", "coils_3d_field_error.png"),
        ("optimized", "coils_3d_field_error_optimized.png"),
    ):
        shot(
            coils[name].plasma.plot.coils_3d(
                surface=log_error[name],
                surface_cmap="magma",
                surface_clim=clim,
            ),
            filename,
            zoom=1.3,
        )
    shot(
        coils["optimized"].plasma.plot.coils_3d(
            color_by="plasma_distance", surface=plasma
        ),
        "coils_3d_distance.png",
        zoom=1.3,
    )
except ImportError as exc:  # pragma: no cover - optional
    print(f"skipped the 3-D views (pyvista unavailable): {exc}")
