"""Generate the figures of the "GVEC equilibria" guide (docs/src/assets/figures/gvec_*) from real
GVEC equilibria.

Like ``generate_real_example_figures.py`` for Struphy, this runs the actual code: GVEC's own
tutorial equilibria (a three-period stellarator and a tokamak, from GVEC's docs), and W7-X from
GVEC's examples, evaluated with ``state.evaluate`` / ``state.evaluate_sfl`` and plotted with
plasma-plots. It needs ``pip install gvec``, which builds GVEC's Fortran core (gfortran, a
LAPACK); the interactive 3-D scenes need PyVista and trame-pyvista, the Plotly charts plotly (see
.github/workflows/docs.yml).

Run from the repo root: python scripts/generate_gvec_figures.py
(or: make figures). W7-X takes a few minutes; set PLASMA_PLOTS_SKIP_W7X=1 to leave it out.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")  # before importing gvec

import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import gvec
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plasma_plots
from plasma_plots.analysis import volume_integral

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "src" / "assets" / "figures"
NUMBERS = DOCS / "src" / "assets" / "gvec" / "numbers.txt"
PLOTLY_OUT = DOCS / "public" / "plotly"
PYVISTA_OUT = DOCS / "public" / "pyvista"
for folder in (OUT, NUMBERS.parent, PLOTLY_OUT, PYVISTA_OUT):
    folder.mkdir(parents=True, exist_ok=True)
WORK = Path(tempfile.mkdtemp(prefix="gvec-runs-"))


def save(result, filename):
    result.save(OUT / filename, close=True)
    print(f"wrote {OUT / filename}")


def save_plotly(result, filename):
    print(f"wrote {result.save(PLOTLY_OUT / f'{filename}.json')}")


def equilibrium(parameters, name):
    """Run GVEC and return its final State."""
    start = time.time()
    run = gvec.run(parameters, runpath=WORK / name, quiet=True)
    print(f"GVEC {name}: {time.time() - start:.1f} s")
    return run.state


# =============================================================================
# The equilibria: GVEC's tutorial stellarator (docs/tutorials 051_plotting, three field periods,
# a rotating ellipse), its tutorial tokamak (010_tokamak, elliptic), and W7-X (examples/).
# =============================================================================
STELLARATOR = {
    "ProjectName": "stellarator",
    "which_hmap": 1,
    "PhiEdge": 1.0,
    "iota": {"type": "polynomial", "coefs": [0.625, 0.35]},
    "pres": {"type": "polynomial", "coefs": [1.0, -1.0], "scale": 1000.0},
    "nfp": 3,
    "X1_b_cos": {(0, 0): 3.0, (1, 0): 1.0, (1, 1): 0.4},
    "X2_b_sin": {(1, 0): 1.0, (1, 1): -0.4, (0, 1): -0.25},
    "init_average_axis": True,
    "sgrid_nElems": 2,
    "X1_mn_max": [3, 3],
    "X2_mn_max": [3, 3],
    "LA_mn_max": [3, 3],
    "X1X2_deg": 5,
    "LA_deg": 5,
    "totalIter": 1000,
    "minimize_tol": 1.0e-6,
}
TOKAMAK = {
    "ProjectName": "tokamak",
    "which_hmap": 1,
    "PhiEdge": 1.0,
    "iota": {"type": "polynomial", "coefs": [0.625, 0.35]},
    "pres": {"type": "interpolation", "rho2": [0.0, 0.25, 0.5, 0.75, 1.0], "vals": [1.0, 0.75, 0.5, 0.25, 0.0],
             "scale": 1000.0},
    "nfp": 1,
    "X1_b_cos": {(0, 0): 5.0, (1, 0): 0.9},
    "X2_b_sin": {(1, 0): 1.1},
    "X1_a_cos": {(0, 0): 5.0},
    "X1_mn_max": [3, 0],
    "X2_mn_max": [3, 0],
    "LA_mn_max": [3, 0],
    "sgrid_nElems": 2,
    "X1X2_deg": 5,
    "LA_deg": 5,
    "totalIter": 10000,
    "minimize_tol": 1e-6,
}

stellarator = equilibrium(STELLARATOR, "stellarator")
nfp = stellarator.nfp
# one field period, on GVEC's logical angles; the geometry (pos) and the grid (X1, X2, theta_P)
# go on every variable with from_gvec
ev = plasma_plots.from_gvec(stellarator.evaluate(
    "mod_B", "pos", "X1", "X2", "Jac", "iota", "p", "theta_P", "N_FP", rho=17, theta=64, zeta=40))
# the same field period on a Boozer grid, on ten flux surfaces
boozer = plasma_plots.from_gvec(stellarator.evaluate_sfl(
    "mod_B", "pos", "N_FP", rho=np.linspace(0.1, 1.0, 10), theta=64, zeta=40, sfl="boozer"))

# =============================================================================
# Poloidal planes
# =============================================================================
lines = {"coordinate_lines": {"rho": 4, "theta_P": 8}}
save(ev.mod_B.plasma.plot.panels(sweep="zeta", coords="physical", plane="RZ", nrows=1, ncols=3, overlays=lines),
     "gvec_poloidal_planes.png")

tokamak = equilibrium(TOKAMAK, "tokamak")
tok = plasma_plots.from_gvec(tokamak.evaluate("mod_B", "pos", "X1", "X2", "iota", "theta_P", "N_FP",
                                              rho=17, theta=64, zeta=[0.0]))
q = (1 / tok.iota).rename("q")
q.attrs = {"label": "$q$", "nfp": 1}
with plasma_plots.figure(1, 2) as fig:
    tok.mod_B.plasma.plot.slice(coords="physical", plane="RZ", zeta=0.0,
                            overlays={"coordinate_lines": {"rho": 5, "theta_P": 12}}, ax=fig[0])
    q.plasma.plot.lineout(rationals=4, ax=fig[1])
save(fig, "gvec_tokamak.png")

# =============================================================================
# On a flux surface: GVEC's own angles and Boozer's, and the logical angles over the Boozer grid
# =============================================================================
with plasma_plots.figure(1, 2) as fig:
    ev.mod_B.plasma.plot.slice(x="zeta", y="theta", rho=0.5, levels=12, ax=fig[0])
    boozer.mod_B.plasma.plot.slice(x="zeta_B", y="theta_B", rho=0.5, levels=12, ax=fig[1])
save(fig, "gvec_surfaces.png")
save(boozer.mod_B.plasma.plot.slice(x="zeta_B", y="theta_B", rho=0.5,
                                    overlays={"coordinate_lines": {"theta": 12, "zeta": 8}}),
     "gvec_boozer_grid.png")

# =============================================================================
# Fourier modes of |B| on the Boozer grid
# =============================================================================
save(boozer.mod_B.plasma.plot.mode_map(rho=0.5, m_range=(-4, 4), n_range=(-9, 9)), "gvec_mode_map.png")
save(boozer.mod_B.plasma.plot.mode_profiles(top=5), "gvec_mode_profiles.png")

# =============================================================================
# Profiles: the rotational transform with its rational surfaces, the pressure, <|B|>, |B| on axis
# =============================================================================
with plasma_plots.figure(2, 2) as fig:
    ev.iota.plasma.plot.lineout(rationals=4, ax=fig[0])
    ev.p.plasma.plot.lineout(ax=fig[1])
    ev.plasma.analysis.surface_average("mod_B").plasma.plot.lineout(ax=fig[2])
    ev.mod_B.plasma.plot.lineout(x="zeta", rho=0, theta=0, ax=fig[3])
save(fig, "gvec_profiles.png")

# =============================================================================
# Numbers: the volume from GVEC's Jacobian on its integration points, against GVEC's own
# =============================================================================
gauss = plasma_plots.from_gvec(stellarator.evaluate("mod_B", "Jac", "pos", "N_FP", rho="int", theta="int", zeta="int"))
volume = float(volume_integral(xr.ones_like(gauss.Jac), jacobian=gauss.Jac)) * gauss.nfp
numerical = float(volume_integral(xr.ones_like(ev.mod_B))) * ev.nfp
gvec_volume = float(stellarator.evaluate("V", rho=[1.0]).V)
average = ev.plasma.analysis.surface_average("mod_B")
NUMBERS.write_text(
    f"plasma volume, GVEC's own V                          {gvec_volume:.10f}\n"
    f"volume_integral(..., jacobian=Jac) * nfp, Gauss grid   {volume:.10f}\n"
    f"volume_integral(...) * nfp, sqrt(g) from X, Y, Z       {numerical:.10f}   (17 x 64 x 40 points)\n"
    f"<|B|> at rho = 0.5                                     {float(average.sel(rho=0.5)):.6f}\n"
)
print(NUMBERS.read_text())

# =============================================================================
# Interactive 3-D: the whole stellarator (every field period), a cutaway and field lines
# =============================================================================
try:
    import pyvista as pv

    pv.OFF_SCREEN = True

    def shot(plotter, filename, *, zoom=1.0, camera="iso"):
        """A PNG, and a standalone interactive HTML scene for <PyVistaScene>."""
        plotter.camera_position = camera
        plotter.reset_camera()
        plotter.camera.zoom(zoom)
        plotter.screenshot(str(OUT / filename), window_size=[1000, 700])
        print(f"wrote {OUT / filename}")
        html = PYVISTA_OUT / filename.replace(".png", ".html")
        for title in list(plotter.scalar_bars.keys()):  # vtk.js would show them 0-255
            plotter.remove_scalar_bar(title)
        try:
            plotter.trame.export_html(str(html))
            print(f"wrote {html}")
        except Exception as exc:  # needs trame-pyvista
            print(f"skipped {html.name} (interactive export unavailable): {exc}")
        plotter.close()

    torus = plasma_plots.from_gvec(stellarator.evaluate(
        "mod_B", "B", "pos", "N_FP", rho=9, theta=48, zeta=np.linspace(0, 2 * np.pi, 32 * nfp, endpoint=False)))
    shot(torus.mod_B.plasma.plot.slices_3d(cuts={"rho": [1.0]}), "gvec_3d_surface.png")
    shot(torus.mod_B.plasma.plot.slices_3d(cuts={"rho": [0.5], "zeta": [0.0, np.pi / 2, np.pi]}), "gvec_3d_cutaway.png")
    axis = torus.isel(rho=0, theta=0, zeta=0)
    shot(torus.B.plasma.plot.streamlines(n_points=40, source_center=(float(axis.X) + 0.35, float(axis.Y), float(axis.Z)),
                                         source_radius=0.25, tube_radius=0.015),
         "gvec_3d_fieldlines.png")
except ImportError as exc:  # pragma: no cover - optional
    print(f"skipped the 3-D views (pyvista unavailable): {exc}")

# =============================================================================
# W7-X (GVEC's examples/parameter-w7x.toml, in scripts/data; five field periods): planes and its last surface
# =============================================================================
if not os.environ.get("PLASMA_PLOTS_SKIP_W7X"):
    from gvec.util import read_parameters

    # a copy of the file: the sdist does not install GVEC's examples
    w7x = equilibrium(read_parameters(ROOT / "scripts" / "data" / "gvec-parameter-w7x.toml"), "w7x")
    planes = plasma_plots.from_gvec(w7x.evaluate("mod_B", "pos", "X1", "X2", "theta_P", "N_FP", rho=13, theta=96,
                                                 zeta=np.linspace(0, np.pi / w7x.nfp, 3)))
    save(planes.mod_B.plasma.plot.panels(sweep="zeta", coords="physical", plane="RZ", nrows=1, ncols=3,
                                         overlays={"coordinate_lines": {"rho": 4, "theta_P": 12}}),
         "gvec_w7x_planes.png")
    try:
        import pyvista as pv  # noqa: F811

        surface = plasma_plots.from_gvec(w7x.evaluate(
            "mod_B", "pos", "N_FP", rho=[1.0], theta=64, zeta=np.linspace(0, 2 * np.pi, 48 * w7x.nfp, endpoint=False)))
        shot(surface.mod_B.plasma.plot.slices_3d(), "gvec_3d_w7x.png", zoom=1.3)
    except ImportError as exc:  # pragma: no cover - optional
        print(f"skipped the W7-X 3-D view (pyvista unavailable): {exc}")

# =============================================================================
# The same plots with Plotly (the folds in the guide)
# =============================================================================
try:
    import plotly  # noqa: F401

    save_plotly(ev.mod_B.plasma.plot.slice(coords="physical", plane="RZ", zeta=0.0, overlays=lines, backend="plotly"),
                "plotly_gvec_poloidal_plane")
    save_plotly(boozer.mod_B.plasma.plot.slice(x="zeta_B", y="theta_B", rho=0.5, levels=12, backend="plotly"),
                "plotly_gvec_boozer_surface")
    save_plotly(ev.iota.plasma.plot.lineout(rationals=4, backend="plotly"), "plotly_gvec_iota")
except ImportError as exc:  # pragma: no cover - optional
    print(f"skipped the Plotly figures (plotly unavailable): {exc}")

plt.close("all")
print("done")
