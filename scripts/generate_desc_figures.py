"""Generate the figures of the "DESC equilibria" guide (docs/src/assets/figures/desc_*) from real
DESC equilibria.

Like ``generate_gvec_figures.py`` for GVEC, this uses the actual code: the example equilibria
that DESC ships (W7-X, the precise quasi-axisymmetric and quasi-helical stellarators, NCSX,
HELIOTRON, ESTELL and the tokamak DSHAPE), evaluated with ``plasma_plots.from_desc`` and
plotted with plasma-plots. It needs ``pip install desc-opt`` (pure Python, on JAX); the
interactive 3-D scenes need PyVista and trame-pyvista, the Plotly charts plotly (see
.github/workflows/docs.yml).

Run from the repo root: python scripts/generate_desc_figures.py (or: make figures).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import desc.examples
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from desc.grid import LinearGrid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import plasma_plots
from plasma_plots.analysis import surface_average, volume_integral

warnings.filterwarnings("ignore", module="desc")  # DESC's notes on JAX, grids and resolutions

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "src" / "assets" / "figures"
NUMBERS = DOCS / "src" / "assets" / "desc" / "numbers.txt"
PLOTLY_OUT = DOCS / "public" / "plotly"
PYVISTA_OUT = DOCS / "public" / "pyvista"
for folder in (OUT, NUMBERS.parent, PLOTLY_OUT, PYVISTA_OUT):
    folder.mkdir(parents=True, exist_ok=True)


def save(result, filename):
    result.save(OUT / filename, close=True)
    print(f"wrote {OUT / filename}")


def save_plotly(result, filename):
    print(f"wrote {result.save(PLOTLY_OUT / f'{filename}.json')}")


def example(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return desc.examples.get(name)


# =============================================================================
# W7-X: one field period on DESC's angles, with the PEST angle for coordinate lines
# =============================================================================
w7x = example("W7-X")
ev = plasma_plots.from_desc(w7x, ["|B|", "iota", "p", "sqrt(g)", "D_Mercier", "theta_PEST"],
                            rho=17, theta=64, zeta=40)
lines = {"coordinate_lines": {"rho": 4, "theta_P": 12}}

# =============================================================================
# Poloidal planes: W7-X over half a field period, and DESC's example library at zeta = 0
# =============================================================================
planes = plasma_plots.from_desc(w7x, ["|B|", "theta_PEST"], rho=13, theta=96, zeta=np.linspace(0, np.pi / w7x.NFP, 3))
save(planes["|B|"].plasma.plot.panels(sweep="zeta", coords="physical", plane="RZ", nrows=1, ncols=3, overlays=lines),
     "desc_w7x_planes.png")

gallery = ["precise_QA", "precise_QH", "NCSX", "HELIOTRON", "ESTELL", "DSHAPE"]
with plasma_plots.figure(2, 3, figsize=(12, 8)) as fig:
    for ax, name in zip(fig, gallery):
        plane = plasma_plots.from_desc(example(name), ["|B|", "theta_PEST"], rho=9, theta=64, zeta=[0.0])
        plane["|B|"].plasma.plot.slice(coords="physical", plane="RZ", zeta=0.0, title=name,
                                       overlays={"coordinate_lines": {"rho": 4, "theta_P": 8}}, ax=ax)
save(fig, "desc_gallery.png")

# =============================================================================
# On a flux surface: quasi-axisymmetry and quasi-helical symmetry, in PEST angles
# =============================================================================
qa, qh = example("precise_QA"), example("precise_QH")
surfaces = {name: plasma_plots.from_desc(eq, "|B|", rho=[0.5, 1.0], theta=64, zeta=64, sfl="pest")
            for name, eq in (("precise_QA", qa), ("precise_QH", qh))}
with plasma_plots.figure(1, 2, figsize=(12, 4.5)) as fig:
    for ax, (name, surface) in zip(fig, surfaces.items()):
        surface["|B|"].plasma.plot.slice(x="zeta", y="theta_P", rho=1.0, levels=12, title=name, ax=ax)
save(fig, "desc_surfaces.png")
with plasma_plots.figure(1, 2, figsize=(12, 4.5)) as fig:
    for ax, (name, surface) in zip(fig, surfaces.items()):
        surface["|B|"].plasma.plot.mode_map(rho=1.0, m_range=(-4, 4), n_range=(-16, 16), ax=ax).ax.set_title(name)
save(fig, "desc_mode_maps.png")

# =============================================================================
# Profiles: W7-X's rotational transform, pressure, Mercier criterion and <|B|>; a tokamak's q
# =============================================================================
with plasma_plots.figure(2, 2) as fig:
    ev.iota.plasma.plot.lineout(rationals=4, ax=fig[0])
    ev.p.plasma.plot.lineout(ax=fig[1])
    ev.D_Mercier.sel(rho=slice(0.1, None)).plasma.plot.lineout(ax=fig[2])  # singular on the axis
    ev["|B|"].plasma.analysis.surface_average(jacobian=ev["sqrt(g)"]).plasma.plot.lineout(ax=fig[3])
save(fig, "desc_profiles.png")

dshape = example("DSHAPE")
tok = plasma_plots.from_desc(dshape, ["|B|", "iota", "theta_PEST"], rho=17, theta=64, zeta=[0.0])
q = (1 / tok.iota).rename("q")
q.attrs = {"label": "$q$", "nfp": 1}
with plasma_plots.figure(1, 2) as fig:
    tok["|B|"].plasma.plot.slice(coords="physical", plane="RZ", zeta=0.0,
                                 overlays={"coordinate_lines": {"rho": 5, "theta_P": 12}}, ax=fig[0])
    q.plasma.plot.lineout(rationals=4, ax=fig[1])
save(fig, "desc_tokamak.png")

# =============================================================================
# Fields: the current density in W7-X's bean-shaped plane
# =============================================================================
fields = plasma_plots.from_desc(w7x, ["|J|", "theta_PEST"], rho=np.linspace(0.05, 1, 20), theta=64, zeta=24)
save(fields["|J|"].plasma.plot.slice(coords="physical", plane="RZ", zeta=0.0, overlays=lines), "desc_current.png")

# =============================================================================
# Numbers: plasma-plots' integrals against DESC's own
# =============================================================================
fine = plasma_plots.from_desc(w7x, ["|B|", "sqrt(g)", "V"], rho=33, theta=64, zeta=48)
volume = float(volume_integral(xr.ones_like(fine["sqrt(g)"]), jacobian=fine["sqrt(g)"])) * w7x.NFP
numerical = float(volume_integral(xr.ones_like(fine["|B|"]))) * w7x.NFP
average = float(surface_average(fine["|B|"].sel(rho=[0.5]), jacobian=fine["sqrt(g)"].sel(rho=[0.5])).squeeze())
desc_average = w7x.compute("<|B|>", grid=LinearGrid(rho=np.array([0.5]), M=32, N=24, NFP=w7x.NFP))["<|B|>"][0]
NUMBERS.write_text(
    f"W7-X volume, DESC's own V                                 {float(fine.V):.10f}\n"
    f"volume_integral(..., jacobian=sqrt(g)) * nfp              {volume:.10f}   (33 x 64 x 48 points)\n"
    f"volume_integral(...) * nfp, sqrt(g) from X, Y, Z          {numerical:.10f}\n"
    f"<|B|> at rho = 0.5, DESC's own                            {desc_average:.10f}\n"
    f"surface_average(|B|, jacobian=sqrt(g)) at rho = 0.5       {average:.10f}\n"
)
print(NUMBERS.read_text())

# =============================================================================
# Interactive 3-D: precise_QH's last surface, a W7-X cutaway and field lines
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
        for title in list(plotter.scalar_bars.keys()):  # vtk.js would show them 0-255
            plotter.remove_scalar_bar(title)
        try:
            plotter.trame.export_html(str(html))
            print(f"wrote {html}")
        except Exception as exc:  # needs trame-pyvista
            print(f"skipped {html.name} (interactive export unavailable): {exc}")
        plotter.close()

    qh_torus = plasma_plots.from_desc(qh, "|B|", rho=[1.0], theta=64,
                                      zeta=np.linspace(0, 2 * np.pi, 32 * qh.NFP, endpoint=False))
    shot(qh_torus["|B|"].plasma.plot.slices_3d(), "desc_3d_qh.png", zoom=1.3)
    torus = plasma_plots.from_desc(w7x, ["|B|", "B"], rho=9, theta=48,
                                   zeta=np.linspace(0, 2 * np.pi, 32 * w7x.NFP, endpoint=False))
    shot(torus["|B|"].plasma.plot.slices_3d(cuts={"rho": [0.5], "zeta": [0.0, np.pi / 2, np.pi]}),
         "desc_3d_cutaway.png", zoom=1.2)
    axis = torus.isel(rho=0, theta=0, zeta=0)
    shot(torus.B.plasma.plot.streamlines(n_points=40, source_center=(float(axis.X) + 0.2, float(axis.Y), float(axis.Z)),
                                         source_radius=0.15, tube_radius=0.02),
         "desc_3d_fieldlines.png", zoom=1.2)
except ImportError as exc:  # pragma: no cover - optional
    print(f"skipped the 3-D views (pyvista unavailable): {exc}")

# =============================================================================
# The same plots with Plotly (the folds in the guide)
# =============================================================================
try:
    import plotly  # noqa: F401

    save_plotly(ev["|B|"].plasma.plot.slice(coords="physical", plane="RZ", zeta=0.0, overlays=lines, backend="plotly"),
                "plotly_desc_poloidal_plane")
    save_plotly(surfaces["precise_QH"]["|B|"].plasma.plot.slice(x="zeta", y="theta_P", rho=1.0, levels=12,
                                                                backend="plotly"),
                "plotly_desc_qh_surface")
    save_plotly(ev.iota.plasma.plot.lineout(rationals=4, backend="plotly"), "plotly_desc_iota")
except ImportError as exc:  # pragma: no cover - optional
    print(f"skipped the Plotly figures (plotly unavailable): {exc}")

plt.close("all")
print("done")
