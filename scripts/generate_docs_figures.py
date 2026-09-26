"""Generate every example figure embedded in the docs (docs/src/assets/figures/).

Every figure comes from a REAL struphy simulation (or a real analytic equilibrium), run right
here -- not synthetic numpy data. This needs the full compiled struphy runtime (see
.github/workflows/docs.yml and CONTRIBUTING.md for the system packages/`struphy compile` step);
a few figures also need the optional ``pyvista``/``plotly``/``scope-profiler`` extras and are
skipped (with a printed message) if those aren't installed.

Run from the repo root: python scripts/generate_docs_figures.py
(or: make figures)
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import struphy_plots  # noqa: F401  (registers .struphy on DataArray/Dataset)
from struphy_plots.arrays import axis_label, value_label
from struphy_plots.output_accessors import OutputPlots
from struphy_plots.plotting import PlotResult, plot_convergence, plot_dispersion

DOCS = Path(__file__).resolve().parents[1] / "docs"
OUT = DOCS / "src" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
# GIFs go in public/, not src/assets/: Astro's image pipeline (which processes
# everything under src/assets referenced from markdown) flattens animated
# GIFs to a single static frame. Files under public/ are served as-is.
PUBLIC_OUT = DOCS / "public" / "figures"
PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
# Plotly figure JSON, fetched client-side by <PlotlyChart>; also served as-is from public/.
PLOTLY_OUT = DOCS / "public" / "plotly"
PLOTLY_OUT.mkdir(parents=True, exist_ok=True)


def save(result, filename):
    path = OUT / filename
    result.save(path, close=True)
    print(f"wrote {path}")


def save_fig(fig, filename):
    path = OUT / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def run_folder():
    return tempfile.mkdtemp()


# =============================================================================
# A real Maxwell (vacuum electromagnetic) simulation: two independent cosine
# modes on components 0 and 1 of the electric field, localized around the
# mid-plane in e3 -- gives genuine (t, e1, e2, e3) structure for every field
# plot, plus real electric/magnetic/total energy scalars.
# =============================================================================
from struphy import DerhamOptions, EnvironmentOptions, Time, domains, grids, perturbations  # noqa: E402
from struphy.models import Maxwell  # noqa: E402
from struphy.simulation.sim import Simulation  # noqa: E402

t0 = time.time()
fields_model = Maxwell()
fields_model.em_fields.e_field.add_perturbation(
    perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(0.5,), comp=0, pfuns=("localize",), pfuns_params=(0.15,))
)
fields_model.em_fields.e_field.add_perturbation(
    perturbations.ModesCosCos(ls=(2,), ms=(1,), amps=(0.3,), comp=1, pfuns=("localize",), pfuns_params=(0.15,))
)
fields_sim = Simulation(
    model=fields_model,
    env=EnvironmentOptions(out_folders=run_folder(), sim_folder="fields"),
    time_opts=Time(dt=0.05, Tend=1.0),
    domain=domains.Cuboid(),
    grid=grids.TensorProductGrid(num_elements=(8, 8, 8)),
    derham_opts=DerhamOptions(degree=(2, 2, 2)),
)
fields_sim.run()
fields_out = fields_sim.output
print(f"[fields] ran in {time.time() - t0:.1f}s")

vector_field = fields_out.evaluate("em_fields/e_field")
field = vector_field.isel(component=0)
field.attrs.setdefault("label", r"$E_x$")

save(field.struphy.plot.slice(x="e1", y="e2", e3=0.5, t="last"), "slice.png")
save(field.struphy.plot.panels(x="e1", y="e2", e3=0.5, nrows=2, ncols=3), "panels.png")
save(field.struphy.plot.lineout(x="e1", t="last", e2=0.5, e3=0.5), "lineout.png")

anim = field.struphy.plot.animation(x="e1", y="e2", e3=0.5, step=2, interval=120)
anim_path = PUBLIC_OUT / "animation.gif"
anim.save(anim_path, writer="pillow", fps=8)
print(f"wrote {anim_path}")

save(
    vector_field.struphy.plot.vector(x="e1", y="e2", components=(0, 1), stride=1, e3=0.5, t="last"),
    "vector.png",
)
save(field.struphy.plot.volume_slices(t="last"), "volume_slices.png")

try:
    import pyvista as pv

    pv.OFF_SCREEN = True
    plotter = field.struphy.plot.volume(cmap="viridis", t="last")
    plotter.camera_position = "iso"
    plotter.screenshot(str(OUT / "volume.png"))
    plotter.close()
    print(f"wrote {OUT / 'volume.png'}")
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped volume.png (PyVista unavailable or headless rendering failed): {exc}")

save(OutputPlots(fields_out).scalars(logy=False), "scalars.png")

electric_energy = fields_out.scalars["electric_energy"]
magnetic_energy = fields_out.scalars["magnetic_energy"]
save(electric_energy.struphy.plot.compare(magnetic_energy, mode="difference"), "compare_difference.png")
save(electric_energy.struphy.plot.compare(magnetic_energy, mode="ratio"), "compare_ratio.png")

norm_t = field.struphy.analysis.norm()
save(norm_t.struphy.plot.lineout(x="t", title="Field norm over time"), "norm.png")

total_energy = fields_out.scalars["total_energy"]
save(
    total_energy.struphy.analysis.drift().struphy.plot.lineout(x="t", title="Drift from the initial value"),
    "drift.png",
)
save(
    total_energy.struphy.analysis.relative_error().struphy.plot.lineout(
        x="t", title="Relative energy conservation error"
    ),
    "relative_error.png",
)


# =============================================================================
# A real analytic equilibrium (no simulation needed): a screw-pinch pressure
# profile, decreasing from the core outward.
# =============================================================================
from struphy.fields_background.equils import ScrewPinch  # noqa: E402

equil = ScrewPinch(a=0.45, R0=2.5, beta=0.6)
equil_domain = domains.HollowCylinder(a1=0.02, a2=0.45, Lz=1.2)
equil.domain = equil_domain

from struphy_plots.plotting import plot_equilibrium_profile, show_equilibrium  # noqa: E402

save(plot_equilibrium_profile(equil, equil_domain), "equilibrium.png")

try:
    import pyvista as pv

    pv.OFF_SCREEN = True
    plotter = show_equilibrium(equil, equil_domain, scalars="p0")
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.3)
    plotter.screenshot(str(OUT / "equilibrium_3d.png"))
    plotter.close()
    print(f"wrote {OUT / 'equilibrium_3d.png'}")
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped equilibrium_3d.png (PyVista unavailable or headless rendering failed): {exc}")


# =============================================================================
# A real kinetic (PIC) simulation: a bump-on-tail electron beam in a uniform
# background, electrostatic Vlasov-Ampere -- genuinely unstable, so the field
# energy really grows and then saturates. Also profiled for real (see below).
# =============================================================================
from struphy import (  # noqa: E402
    BinningPlot,
    BoundaryParameters,
    LoadingParameters,
    SavingParameters,
    SortingParameters,
    WeightsParameters,
    maxwellians,
)
from struphy.kinetic_background.base import SumKineticBackground  # noqa: E402
from struphy.models import VlasovAmpereOneSpecies  # noqa: E402

t0 = time.time()
domain_length = 12.56  # ~4 pi: a few wavelengths of the seeded k=1 mode
kinetic_model = VlasovAmpereOneSpecies(alpha=1.0, epsilon=-1.0, with_B0=False)
kinetic_model.kinetic_ions.set_markers(
    loading_params=LoadingParameters(ppc=200, seed=1234),
    weights_params=WeightsParameters(control_variate=True),
    boundary_params=BoundaryParameters(),
    sorting_params=SortingParameters(boxes_per_dim=(8, 1, 1), do_sort=True),
    saving_params=SavingParameters(
        n_markers=24,
        binning_plots=(BinningPlot(slice="e1_v1", n_bins=(48, 48), ranges=((0.0, 1.0), (-5.0, 9.0))),),
    ),
    bufsize=0.4,
)
kinetic_model.propagators.push_eta.options = kinetic_model.propagators.push_eta.Options()
kinetic_model.propagators.coupling_va.options = kinetic_model.propagators.coupling_va.Options()
kinetic_model.initial_poisson.options = kinetic_model.initial_poisson.Options(stab_mat="M0")

beam = maxwellians.Maxwellian3D(n=(0.15, None), u1=(4.0, None), vth1=(0.3, None))
kinetic_model.kinetic_ions.var.add_background(SumKineticBackground(maxwellians.Maxwellian3D(n=(0.85, None)), beam))
kinetic_model.kinetic_ions.var.add_initial_condition(
    SumKineticBackground(maxwellians.Maxwellian3D(n=(1.0, perturbations.ModesCos(ls=(1,), amps=(1e-2,)))), beam)
)

kinetic_sim = Simulation(
    model=kinetic_model,
    env=EnvironmentOptions(out_folders=run_folder(), sim_folder="bump_on_tail"),
    time_opts=Time(dt=0.05, Tend=20.0),
    domain=domains.Cuboid(r1=domain_length),
    grid=grids.TensorProductGrid(num_elements=(16, 1, 1)),
    derham_opts=DerhamOptions(degree=(3, 1, 1)),
)
kinetic_sim.run(profiling_activated=True)
kinetic_out = kinetic_sim.output
print(f"[kinetic] ran in {time.time() - t0:.1f}s")

# A single probe point's raw (oscillating) field: its amplitude grows during
# the linear instability, then saturates and partly relaxes -- real growth
# *and* real damping/envelope behaviour from the same signal.
probe = kinetic_out.evaluate("em_fields/e_field", component=0).isel(e2=0, e3=0, e1=2)
probe.attrs.setdefault("label", r"$E_x$ probe")

save(probe.struphy.plot.timeseries(fit=(1.0, 8.0), title="Bump-on-tail instability: field growth"), "timeseries_growth.png")

envelope = probe.struphy.analysis.envelope()
fit = probe.struphy.analysis.damping_rate(window=(9.0, 13.0))
fig, ax = plt.subplots()
ax.plot(probe.t, probe, lw=0.8, label="probe field")
ax.plot(envelope.t, envelope, "o", ms=3, color="C1", label="envelope peaks")
if fit is not None:
    ax.plot(fit.time, fit.fitted, "--", color="C2", label=rf"fit: $\gamma$ = {fit.rate:.3f}")
ax.set(xlabel=axis_label(probe, "t"), ylabel=value_label(probe), title="Relaxation after saturation")
ax.legend(fontsize="small")
save_fig(fig, "damping.png")

distribution = kinetic_out.evaluate("kinetic_ions/f", dataset="e1_v1_density/f")
save(distribution.struphy.plot.slice(x="e1", y="v1", t="last"), "phase_space.png")

moments = distribution.struphy.analysis.velocity_moments()
final = moments.isel(t=-1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
for ax, name in zip(axes, ("density", "mean_v1", "variance_v1")):
    array = final[name]
    ax.plot(array.e1, array)
    ax.set(xlabel=axis_label(array, "e1"), ylabel=value_label(array), title=array.attrs.get("label", name))
save_fig(fig, "velocity_moments.png")

averaged = distribution.struphy.analysis.spatial_average()
save(
    averaged.struphy.plot.slice(x="t", y="v1", title="Velocity distribution averaged over space"),
    "spatial_average.png",
)

orbits = kinetic_out.evaluate("kinetic_ions/orbits")
save(orbits.struphy.plot.trajectories(show_paths=True, max_markers=8), "trajectories.png")
save(orbits.struphy.plot.scatter(x="x", y="v1", color="weight", t="last"), "marker_scatter.png")

# The same markers, in the same (e1, v1) phase space as the distribution
# above: real individual trajectories over the real phase-space density.
phase_space_orbits = orbits.assign(e1=orbits["x"] / domain_length)
save(
    distribution.struphy.plot.overlay_orbits(phase_space_orbits, x="e1", y="v1", max_markers=6, t="last"),
    "orbit_overlay.png",
)


# =============================================================================
# Diagnostics: a real convergence study -- the L2 projection error of a
# Maxwell mode's initial condition against its exact analytic profile, at
# increasing resolution (no time-stepping needed: Tend=0).
# =============================================================================
def exact_mode(x, y):
    return 0.5 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)


conv_sizes, conv_errors = [], []
for nel in (4, 8, 16, 32):
    model = Maxwell()
    model.em_fields.e_field.add_perturbation(perturbations.ModesCosCos(ls=(1,), ms=(1,), amps=(0.5,), comp=0))
    sim = Simulation(
        model=model,
        env=EnvironmentOptions(out_folders=run_folder(), sim_folder="conv"),
        time_opts=Time(dt=0.01, Tend=0.0),
        domain=domains.Cuboid(),
        grid=grids.TensorProductGrid(num_elements=(nel, nel, 1)),
        derham_opts=DerhamOptions(degree=(2, 2, 1)),
    )
    sim.run()
    da = sim.output.evaluate("em_fields/e_field", component=0).isel(t=0, e3=0)
    error = float(np.sqrt(np.mean((da.values - exact_mode(da.X.values, da.Y.values)) ** 2)))
    conv_sizes.append(nel)
    conv_errors.append(error)
    print(f"[convergence] nel={nel} error={error:.3e}")

fig, ax = plt.subplots()
plot_convergence(conv_sizes, conv_errors, ax=ax, label="Maxwell mode projection")
save_fig(fig, "convergence.png")


# =============================================================================
# Diagnostics: a real dispersion relation, from the kinetic run's raw
# electric field, compared with the (approximate, single-Maxwellian) Bohm-Gross
# branch -- the bump-on-tail background is not a pure Maxwellian, so this is a
# reference curve, not an exact fit.
# =============================================================================
def bohm_gross(k):
    return np.sqrt(1.0 + 3.0 * k**2)


dispersive_field = kinetic_out.evaluate("em_fields/e_field", component=0).isel(e2=0, e3=0)
# "e1" is the logical coordinate (0..1); swap in the physical position so the FFT's
# wavenumber comes out in physical units, matching the Bohm-Gross reference curve.
dispersive_field = dispersive_field.assign_coords(e1=("e1", dispersive_field["X"].values))
save(
    dispersive_field.struphy.plot.dispersion(branches={"Bohm-Gross (approx.)": bohm_gross}, kmax=3, omega_max=8),
    "dispersion.png",
)


# =============================================================================
# Plotly examples via .struphy.data (needs `pip install plotly`): figure JSON
# for the docs' <PlotlyChart> component (docs/src/components/PlotlyChart.astro),
# which loads Plotly.js from a CDN and renders it client-side.
# =============================================================================
try:
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    def save_plotly(fig, filename, *, width=560, height=440):
        # Only the JSON is used (fetched client-side by <PlotlyChart>, which sets its own
        # size); width/height here are just so the figure looks reasonable if opened directly.
        fig.update_layout(width=width, height=height)
        fig.write_json(PLOTLY_OUT / f"{filename}.json")
        print(f"wrote {PLOTLY_OUT / f'{filename}.json'}")

    selected = field.struphy.data.slice(x="e1", y="e2", e3=0.5, t="last")
    fig = px.imshow(
        selected.transpose("e2", "e1"), x=selected.e1, y=selected.e2, origin="lower", color_continuous_scale="viridis"
    )
    fig.update_layout(xaxis_title="e1", yaxis_title="e2")
    save_plotly(fig, "plotly_slice")

    frame = orbits.struphy.data.scatter(x="x", y="v1", color="weight", t="last").to_dataframe()
    fig = px.scatter(frame, x="x", y="v1", color="weight", color_continuous_scale="viridis")
    save_plotly(fig, "plotly_scatter")

    series = probe.struphy.data.timeseries()
    fig = go.Figure()
    for item in series:
        fig.add_trace(go.Scatter(x=item.t, y=item, mode="lines", name=item.attrs.get("label", item.name)))
    fig.update_layout(xaxis_title="t", yaxis_title=value_label(probe), legend=dict(x=0.02, y=0.98))
    save_plotly(fig, "plotly_timeseries")

    vec = vector_field.struphy.data.vector(x="e1", y="e2", components=(0, 1), stride=1, e3=0.5, t="last")
    xg, yg = np.meshgrid(vec.e1.values, vec.e2.values, indexing="ij")
    fig = ff.create_quiver(
        xg.ravel(),
        yg.ravel(),
        vec.isel(component=0).values.ravel(),
        vec.isel(component=1).values.ravel(),
        scale=0.05,
    )
    fig.update_layout(xaxis_title="e1", yaxis_title="e2")
    save_plotly(fig, "plotly_vector")

    planes = field.struphy.data.volume_slices(t="last")
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(planes))
    for i, (name, plane) in enumerate(planes.items(), start=1):
        x, y = plane.dims
        fig.add_trace(
            go.Heatmap(z=plane.values.T, x=plane[x].values, y=plane[y].values, colorscale="viridis", showscale=False),
            row=1,
            col=i,
        )
    save_plotly(fig, "plotly_volume_slices", width=900, height=340)

    subset = orbits.struphy.data.trajectories(max_markers=24)
    fig = go.Figure()
    for marker in subset.marker.values:
        path = subset.sel(marker=marker)
        fig.add_trace(go.Scatter3d(x=path.x, y=path.y, z=path.z, mode="lines", line=dict(width=3), showlegend=False))
    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
    save_plotly(fig, "plotly_trajectories", height=520)

    phase_space_selected = distribution.struphy.data.slice(x="e1", y="v1", t="last")
    fig = px.imshow(
        phase_space_selected.transpose("v1", "e1"),
        x=phase_space_selected.e1,
        y=phase_space_selected.v1,
        origin="lower",
        color_continuous_scale="viridis",
    )
    fig.update_layout(xaxis_title="e1", yaxis_title="v1")
    save_plotly(fig, "plotly_phase_space")

    result = electric_energy.struphy.data.compare(magnetic_energy, mode="ratio")
    fig = px.line(x=result.t, y=result, labels={"x": "t", "y": result.name})
    save_plotly(fig, "plotly_compare")

    field_slice, orbit_subset = distribution.struphy.data.overlay_orbits(phase_space_orbits, x="e1", y="v1", max_markers=6, t="last")
    fig = go.Figure(
        go.Heatmap(
            z=field_slice.transpose("v1", "e1").values,
            x=field_slice.e1.values,
            y=field_slice.v1.values,
            colorscale="viridis",
            showscale=False,
        )
    )
    for marker in orbit_subset.marker.values:
        path = orbit_subset.sel(marker=marker)
        fig.add_trace(go.Scatter(x=path.e1, y=path.v1, mode="lines", line=dict(color="#ffb347"), showlegend=False))
    fig.update_layout(xaxis_title="e1", yaxis_title="v1")
    save_plotly(fig, "plotly_overlay_orbits")

    disp_spectrum = dispersive_field.struphy.data.dispersion()
    disp_values = np.log10(np.asarray(disp_spectrum) + np.finfo(float).tiny)
    positive = disp_spectrum.omega.values >= 0
    k_line = np.linspace(0, 3, 60)
    fig = go.Figure(
        go.Heatmap(
            z=disp_values[positive],
            x=disp_spectrum.k.values,
            y=disp_spectrum.omega.values[positive],
            colorscale="viridis",
            zmin=disp_values.max() - 6,
            zmax=disp_values.max(),
            showscale=False,
        )
    )
    fig.add_trace(
        go.Scatter(x=k_line, y=bohm_gross(k_line), mode="lines", name="Bohm-Gross (approx.)", line=dict(color="#ffb347", dash="dash"))
    )
    fig.update_layout(xaxis_title="k", yaxis_title="omega", xaxis_range=[-3, 3], yaxis_range=[0, 8])
    save_plotly(fig, "plotly_dispersion")
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped plotly_*.json (plotly unavailable): {exc}")


# =============================================================================
# Whole-run: profiling, from the REAL kinetic run above (profiling_activated=True)
# =============================================================================
try:
    profile_plots = OutputPlots(kinetic_out).profile
    save(PlotResult(*profile_plots.gantt(include=["prop:", "kernel:", "solve:"])), "profile_gantt.png")
    region_filter = ["prop:", "kernel:", "solve:"]
    save(PlotResult(*profile_plots.flame(include=region_filter)), "profile_flame.png")
    save(PlotResult(*profile_plots.callgraph(compact=True, include=region_filter)), "profile_callgraph.png")

    try:
        import plotly  # noqa: F401

        fig = profile_plots.gantt(backend="plotly", include=["prop:", "kernel:", "solve:"])
        fig.write_json(PLOTLY_OUT / "plotly_profile_gantt.json")
        print(f"wrote {PLOTLY_OUT / 'plotly_profile_gantt.json'}")
    except ImportError:
        print("skipped plotly_profile_gantt.json (plotly unavailable)")
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped profile_*.png (scope-profiler unavailable): {exc}")

plt.close("all")
print("done")
