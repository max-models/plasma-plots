"""Generate every general-purpose example figure embedded in the docs
(docs/src/assets/figures/).

Builds small, purely synthetic labeled xarray data and renders it with struphy_plots,
so the docs show real output of the actual plotting/analysis code rather than mockups,
without needing a struphy install at all. Most figures need only struphy_plots +
numpy/xarray/matplotlib; a few (marked below) need the optional ``pyvista`` extra
(``pip install -e ".[pyvista]"``) and a working off-screen rendering setup (see
``.github/workflows/docs.yml``).

The one exception is the "A real simulation" guide, whose figures (``real_*.png``,
``real_plotly_*.json``) come from an actual struphy run -- see
``generate_real_example_figures.py``.

Run from the repo root: python scripts/generate_docs_figures.py
(or: make figures, which runs this and generate_real_example_figures.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import struphy_plots  # noqa: F401  (registers .struphy on DataArray/Dataset)
from struphy_plots.arrays import axis_label, value_label
from struphy_plots.plotting import PlotResult, plot_convergence, plot_dispersion, plot_scalars

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


def field_array(name, label, unit, values, dims, coords):
    return xr.DataArray(
        values,
        dims=dims,
        coords=coords,
        name=name,
        attrs={"label": label, "units": unit},
    )


def save(result, filename):
    path = OUT / filename
    result.save(path, close=True)
    print(f"wrote {path}")


def save_fig(fig, filename):
    path = OUT / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# =============================================================================
# Field plots: a wave packet orbiting the unit square
# =============================================================================
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

field = field_array("phi", r"$\phi$", "a.u.", phi, ("t", "e1", "e2"), {"t": t, "e1": e1, "e2": e2})

save(field.struphy.plot.slice(x="e1", y="e2", t="last"), "slice.png")
save(field.struphy.plot.panels(x="e1", y="e2", nrows=2, ncols=3), "panels.png")
save(field.struphy.plot.lineout(x="e1", t="last", e2=0.5), "lineout.png")

anim = field.struphy.plot.animation(x="e1", y="e2", step=2, interval=120)
anim_path = PUBLIC_OUT / "animation.gif"
anim.save(anim_path, writer="pillow", fps=8)
print(f"wrote {anim_path}")

# A solid-body rotation vector field on the same grid
vx = -(E2 - 0.5)
vy = E1 - 0.5
vector = field_array(
    "v",
    r"$\mathbf{v}$",
    "a.u.",
    np.stack(
        [
            np.broadcast_to(vx, (n_t, n_e1, n_e2)),
            np.broadcast_to(vy, (n_t, n_e1, n_e2)),
        ],
        axis=1,
    ),
    ("t", "component", "e1", "e2"),
    {"t": t, "e1": e1, "e2": e2},
)
save(
    vector.struphy.plot.vector(x="e1", y="e2", components=(0, 1), stride=6, t="last"),
    "vector.png",
)

# A 3-D scalar blob for the orthogonal-slices and volume renders
n3 = 40
e1_3, e2_3, e3_3 = (np.linspace(0.0, 1.0, n3) for _ in range(3))
E1_3, E2_3, E3_3 = np.meshgrid(e1_3, e2_3, e3_3, indexing="ij")
blob = np.exp(-((E1_3 - 0.5) ** 2 + (E2_3 - 0.5) ** 2 + (E3_3 - 0.5) ** 2) / (2 * 0.15**2))
volume_data = field_array("n", "$n$", "a.u.", blob, ("e1", "e2", "e3"), {"e1": e1_3, "e2": e2_3, "e3": e3_3})
save(volume_data.struphy.plot.volume_slices(), "volume_slices.png")

try:
    import pyvista as pv

    pv.OFF_SCREEN = True

    physical = volume_data.assign_coords(
        X=(("e1", "e2", "e3"), E1_3),
        Y=(("e1", "e2", "e3"), E2_3),
        Z=(("e1", "e2", "e3"), E3_3),
    )
    plotter = physical.struphy.plot.volume(cmap="viridis")
    plotter.camera_position = "iso"
    plotter.screenshot(str(OUT / "volume.png"))
    plotter.close()
    print(f"wrote {OUT / 'volume.png'}")
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped volume.png (PyVista unavailable or headless rendering failed): {exc}")


# =============================================================================
# Time series, growth/damping rates, and run comparisons
# =============================================================================
tt = np.linspace(0.0, 6.0, 200)
rate = 0.6
en_phi = np.exp(rate * tt) / (1.0 + 0.02 * np.exp(rate * tt)) + 1e-3 * np.cos(20 * tt)
en_tot = 1.0 + 0.02 * (1.0 - np.exp(-0.5 * tt)) + 2e-3 * np.sin(15 * tt)

energy = field_array("en_phi", r"$e_\phi$", "J", en_phi, ("t",), {"t": tt})
total = field_array("en_tot", r"$e_{tot}$", "J", en_tot, ("t",), {"t": tt})

save(
    energy.struphy.plot.timeseries(fit=(0.0, 2.0), title="Field energy growth"),
    "timeseries_growth.png",
)
save(plot_scalars({"en_phi": energy, "en_tot": total}, logy=False), "scalars.png")


def oscillating_energy(rate=-0.3, omega=3.0):
    time = np.linspace(0.0, 20.0, 4001)
    values = np.exp(2 * rate * time) * np.cos(omega * time) ** 2 + 1e-12
    return field_array("energy", r"$e$", "J", values, ("t",), {"t": time})


damped = oscillating_energy()
envelope = damped.struphy.analysis.envelope()
fit = damped.struphy.analysis.damping_rate(amplitude=True)
fig, ax = plt.subplots()
ax.plot(damped.t, damped, lw=0.8, label="energy")
ax.plot(envelope.t, envelope, "o", ms=3, color="C1", label="envelope peaks")
ax.plot(fit.time, fit.fitted, "--", color="C2", label=rf"fit: $\gamma$ = {fit.rate:.3f}")
ax.set(
    xlabel=axis_label(damped, "t"),
    ylabel=value_label(damped),
    title="Damping rate from the envelope",
)
ax.legend(fontsize="small")
save_fig(fig, "damping.png")

run_a = field_array("en_phi", r"$e_\phi$", "J", np.exp(0.55 * tt), ("t",), {"t": tt})
run_b = field_array("en_phi", r"$e_\phi$", "J", np.exp(0.62 * tt), ("t",), {"t": tt})
save(run_a.struphy.plot.compare(run_b, mode="difference"), "compare_difference.png")
save(run_a.struphy.plot.compare(run_b, mode="ratio"), "compare_ratio.png")


# =============================================================================
# Diagnostics: norm, drift, relative error
# =============================================================================
decaying_field = field * xr.DataArray(1.0 / (1.0 + 0.4 * t), dims=("t",), coords={"t": t})
decaying_field.attrs = dict(field.attrs)
norm_t = decaying_field.struphy.analysis.norm()
save(norm_t.struphy.plot.lineout(x="t", title="Field norm decaying in time"), "norm.png")

save(
    total.struphy.analysis.drift().struphy.plot.lineout(x="t", title="Drift from the initial value"),
    "drift.png",
)

en_cons = field_array(
    "en_tot",
    r"$e_{tot}$",
    "J",
    1.0 + 2e-4 * tt + 3e-5 * np.sin(30 * tt),
    ("t",),
    {"t": tt},
)
save(
    en_cons.struphy.analysis.relative_error().struphy.plot.lineout(x="t", title="Relative energy conservation error"),
    "relative_error.png",
)


# =============================================================================
# Particles: a growing bump-on-tail distribution
# =============================================================================
n_tp, n_e1p, n_v1p = 20, 48, 160
tp = np.linspace(0.0, 5.0, n_tp)
e1p = np.linspace(0.0, 1.0, n_e1p)
v1p = np.linspace(-4.0, 4.0, n_v1p)
TP, E1P, V1P = np.meshgrid(tp, e1p, v1p, indexing="ij")

bulk = np.exp(-(V1P**2) / 2.0)
beam_amplitude = 0.1 + 0.7 * TP / tp[-1]
beam = beam_amplitude * np.exp(-((V1P - 3.0) ** 2) / (2 * 0.4**2))
f = (bulk + beam) * (1.0 + 0.15 * np.cos(2 * np.pi * E1P))

distribution = field_array("f", "$f$", "a.u.", f, ("t", "e1", "v1"), {"t": tp, "e1": e1p, "v1": v1p})

save(distribution.struphy.plot.slice(x="e1", y="v1", t="last"), "phase_space.png")

moments = distribution.struphy.analysis.velocity_moments()
final = moments.isel(t=-1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
for ax, name in zip(axes, ("density", "mean_v1", "variance_v1")):
    array = final[name]
    ax.plot(array.e1, array)
    ax.set(
        xlabel=axis_label(array, "e1"),
        ylabel=value_label(array),
        title=array.attrs.get("label", name),
    )
save_fig(fig, "velocity_moments.png")

averaged = distribution.struphy.analysis.spatial_average()
save(
    averaged.struphy.plot.slice(x="t", y="v1", title="Velocity distribution averaged over space"),
    "spatial_average.png",
)


# =============================================================================
# Marker trajectories (the per-quantity orbits Dataset)
# =============================================================================
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
    {"x": (("t", "marker"), X), "y": (("t", "marker"), Y), "z": (("t", "marker"), Z)},
    coords={"t": s, "marker": np.arange(n_markers)},
    attrs={"product": "orbits", "label": "marker orbits"},
)
save(orbits.struphy.plot.trajectories(show_paths=True), "trajectories.png")

# A cloud of Lagrangian particles (e.g. an SPH gas expansion), colored by a
# tracer they carry -- their own initial radius.
n_particles = 400
rng_p = np.random.default_rng(2)
angle = rng_p.uniform(0, 2 * np.pi, n_particles)
radius0 = rng_p.rayleigh(0.15, n_particles)
tracer = radius0.copy()
expansion = 2.2
x_p = expansion * radius0 * np.cos(angle)
y_p = expansion * radius0 * np.sin(angle)
cloud = xr.Dataset(
    {
        "x": ("marker", x_p),
        "y": ("marker", y_p),
        "density": ("marker", np.exp(-(radius0**2) / (2 * 0.15**2))),
    },
    coords={"marker": np.arange(n_particles)},
    attrs={"label": "Expanding particle cloud"},
)
save(cloud.struphy.plot.scatter(x="x", y="y", color="density"), "marker_scatter.png")

# Marker orbits overlaid on a background field (a Poincare-style diagnostic):
# a potential well with a few near-circular confined orbits at different radii.
n_bg = 96
e1_bg, e2_bg = np.linspace(0.0, 1.0, n_bg), np.linspace(0.0, 1.0, n_bg)
E1_BG, E2_BG = np.meshgrid(e1_bg, e2_bg, indexing="ij")
well = field_array(
    "phi",
    r"$\phi$",
    "a.u.",
    np.exp(-((E1_BG - 0.5) ** 2 + (E2_BG - 0.5) ** 2) / (2 * 0.2**2)),
    ("e1", "e2"),
    {"e1": e1_bg, "e2": e2_bg},
)
n_orbit_steps, n_confined = 60, 6
s_orbit = np.linspace(0.0, 2 * np.pi, n_orbit_steps)
radii_orbit = np.linspace(0.15, 0.3, n_confined)
phases_orbit = rng_p.uniform(0, 2 * np.pi, n_confined)
orbit_e1 = 0.5 + radii_orbit[None, :] * np.cos(s_orbit[:, None] + phases_orbit[None, :])
orbit_e2 = 0.5 + radii_orbit[None, :] * np.sin(s_orbit[:, None] + phases_orbit[None, :])
confined_orbits = xr.Dataset(
    {"e1": (("t", "marker"), orbit_e1), "e2": (("t", "marker"), orbit_e2)},
    coords={"t": s_orbit, "marker": np.arange(n_confined)},
)
save(
    well.struphy.plot.overlay_orbits(confined_orbits, x="e1", y="e2"),
    "orbit_overlay.png",
)


# =============================================================================
# Diagnostics: a convergence study
# =============================================================================
sizes = np.array([8.0, 16.0, 32.0, 64.0, 128.0])
first_order = 0.4 / sizes
second_order = 0.4 / sizes**2
fig, ax = plt.subplots()
plot_convergence(sizes, first_order, ax=ax, label="scheme A")
plot_convergence(sizes, second_order, ax=ax, label="scheme B")
ax.legend(fontsize="small")
save_fig(fig, "convergence.png")


# =============================================================================
# Diagnostics: a dispersion relation
# =============================================================================
# A few Langmuir-like modes obeying the Bohm-Gross relation omega^2 = 1 + 3k^2,
# each launched as a standing wave (both +k and -k components), plus noise.
n_t_disp, n_x_disp, length = 500, 128, 2 * np.pi
t_disp = np.linspace(0.0, 60.0, n_t_disp)
x_disp = np.linspace(0.0, length, n_x_disp, endpoint=False)
X_DISP, T_DISP = np.meshgrid(x_disp, t_disp)


def bohm_gross(k):
    return np.sqrt(1.0 + 3.0 * k**2)


rng_disp = np.random.default_rng(3)
wave = np.zeros_like(X_DISP)
for k in (2.0, 3.0, 4.0, 5.0):
    omega_k = bohm_gross(k)
    wave += np.cos(k * X_DISP - omega_k * T_DISP) + np.cos(k * X_DISP + omega_k * T_DISP)
wave += 0.05 * rng_disp.standard_normal(wave.shape)

dispersive_field = field_array("phi", r"$\phi$", "a.u.", wave, ("t", "e1"), {"t": t_disp, "e1": x_disp})
save(
    dispersive_field.struphy.plot.dispersion(branches={"Bohm-Gross": bohm_gross}, kmax=7, omega_max=12),
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

    selected = field.struphy.data.slice(x="e1", y="e2", t="last")
    fig = px.imshow(
        selected.transpose("e2", "e1"),
        x=selected.e1,
        y=selected.e2,
        origin="lower",
        color_continuous_scale="viridis",
    )
    fig.update_layout(xaxis_title="e1", yaxis_title="e2")
    save_plotly(fig, "plotly_slice")

    frame = cloud.struphy.data.scatter(x="x", y="y", color="density").to_dataframe()
    fig = px.scatter(frame, x="x", y="y", color="density", color_continuous_scale="viridis")
    save_plotly(fig, "plotly_scatter")

    series = energy.struphy.data.timeseries(total)
    fig = go.Figure()
    for item in series:
        fig.add_trace(go.Scatter(x=item.t, y=item, mode="lines", name=item.attrs.get("label", item.name)))
    fig.update_layout(xaxis_title="t", yaxis_title="[J]", legend=dict(x=0.02, y=0.98))
    save_plotly(fig, "plotly_timeseries")

    vec = vector.struphy.data.vector(x="e1", y="e2", components=(0, 1), stride=4, t="last")
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

    planes = volume_data.struphy.data.volume_slices()
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(planes))
    for i, (name, plane) in enumerate(planes.items(), start=1):
        x, y = plane.dims
        fig.add_trace(
            go.Heatmap(
                z=plane.values.T,
                x=plane[x].values,
                y=plane[y].values,
                colorscale="viridis",
                showscale=False,
            ),
            row=1,
            col=i,
        )
    save_plotly(fig, "plotly_volume_slices", width=900, height=340)

    subset = orbits.struphy.data.trajectories(max_markers=24)
    fig = go.Figure()
    for marker in subset.marker.values:
        path = subset.sel(marker=marker)
        fig.add_trace(
            go.Scatter3d(
                x=path.x,
                y=path.y,
                z=path.z,
                mode="lines",
                line=dict(width=3),
                showlegend=False,
            )
        )
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

    result = run_a.struphy.data.compare(run_b, mode="ratio")
    fig = px.line(x=result.t, y=result, labels={"x": "t", "y": result.name})
    save_plotly(fig, "plotly_compare")

    field_slice, orbit_subset = well.struphy.data.overlay_orbits(confined_orbits, x="e1", y="e2")
    fig = go.Figure(
        go.Heatmap(
            z=field_slice.transpose("e2", "e1").values,
            x=field_slice.e1.values,
            y=field_slice.e2.values,
            colorscale="viridis",
            showscale=False,
        )
    )
    for marker in orbit_subset.marker.values:
        path = orbit_subset.sel(marker=marker)
        fig.add_trace(
            go.Scatter(
                x=path.e1,
                y=path.e2,
                mode="lines",
                line=dict(color="#ffb347"),
                showlegend=False,
            )
        )
    fig.update_layout(xaxis_title="e1", yaxis_title="e2")
    save_plotly(fig, "plotly_overlay_orbits")

    disp_spectrum = dispersive_field.struphy.data.dispersion()
    disp_values = np.log10(np.asarray(disp_spectrum) + np.finfo(float).tiny)
    positive = disp_spectrum.omega.values >= 0
    k_line = np.linspace(0, 7, 60)
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
        go.Scatter(
            x=k_line,
            y=bohm_gross(k_line),
            mode="lines",
            name="Bohm-Gross",
            line=dict(color="#ffb347", dash="dash"),
        )
    )
    fig.update_layout(xaxis_title="k", yaxis_title="omega", xaxis_range=[-7, 7], yaxis_range=[0, 12])
    save_plotly(fig, "plotly_dispersion")
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped plotly_*.json (plotly unavailable): {exc}")


# =============================================================================
# Whole-run: equilibrium profiles (optional PyVista)
#
# plot_equilibrium_profile/show_equilibrium take a struphy-shaped equilibrium
# (``.p0(e1, e2, e3)``, optionally ``.n0(...)``) and domain mapping
# (``domain(e1, e2, e3, squeeze_out=False) -> (X, Y, Z)``) -- normally
# ``out.equil``/``out.domain`` from a real run. These small duck-typed
# stand-ins keep this figure synthetic without a struphy install.
# =============================================================================
try:
    import pyvista as pv

    from struphy_plots.plotting import plot_equilibrium_profile, show_equilibrium

    pv.OFF_SCREEN = True

    class RadialEquil:
        """A pressure/density profile decreasing from the core outward."""

        def p0(self, e1, e2, e3):
            r = np.meshgrid(e1, e2, e3, indexing="ij")[0]
            return (1 - r**2) ** 1.5 + 0.05

        def n0(self, e1, e2, e3):
            r = np.meshgrid(e1, e2, e3, indexing="ij")[0]
            return (1 - r**2) + 0.1

    def radial_domain(e1, e2, e3, squeeze_out=True):
        """A trivial straight radial line: e1 *is* the physical radius."""
        r = np.meshgrid(e1, e2, e3, indexing="ij")[0]
        return r, np.zeros_like(r), np.zeros_like(r)

    save(plot_equilibrium_profile(RadialEquil(), radial_domain), "equilibrium.png")

    class ShellEquil:
        """A poloidally-varying pressure on a toroidal boundary shell."""

        def p0(self, e1, e2, e3):
            theta = 2 * np.pi * np.meshgrid(e1, e2, e3, indexing="ij")[1]
            return 1.0 + 0.6 * np.cos(2 * theta)

    def shell_domain(e1, e2, e3, squeeze_out=True):
        """A toroidal shell, thickened slightly along e1 to avoid a degenerate grid."""
        E1, E2, E3 = np.meshgrid(e1, e2, e3, indexing="ij")
        r_minor, R0 = 0.35 * (0.85 + 0.15 * E1), 1.0
        theta, phi_ = 2 * np.pi * E2, 2 * np.pi * E3
        x = (R0 + r_minor * np.cos(theta)) * np.cos(phi_)
        y = (R0 + r_minor * np.cos(theta)) * np.sin(phi_)
        z = r_minor * np.sin(theta)
        return x, y, z

    plotter = show_equilibrium(ShellEquil(), shell_domain, scalars="p0")
    plotter.camera_position = "iso"
    plotter.camera.zoom(1.3)
    plotter.screenshot(str(OUT / "equilibrium_3d.png"))
    plotter.close()
    print(f"wrote {OUT / 'equilibrium_3d.png'}")
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped equilibrium figures (PyVista unavailable or headless rendering failed): {exc}")


# =============================================================================
# Whole-run: profiling (optional scope-profiler)
# =============================================================================
try:
    import tempfile as _tempfile
    import time as _time

    import scope_profiler as _sp

    from struphy_plots.output_accessors import OutputPlots

    class _FakeProfile:
        def __init__(self, results):
            self.results = results

    class _FakeOutput:
        """Anything with a `.profile.results` (a scope_profiler.ProfilingResults) works."""

        def __init__(self, results):
            self.profile = _FakeProfile(results)

    _profiling_tmp = _tempfile.mkdtemp()
    with _sp.session(verbose=False, file_path=str(Path(_profiling_tmp) / "profiling_data.h5")):
        for _ in range(6):
            with _sp.region("prop: faraday"):
                _time.sleep(0.001)
            with _sp.region("prop: push_eta"):
                with _sp.region("kernel: evaluate"):
                    _time.sleep(0.002)
                with _sp.region("kernel: interpolate"):
                    _time.sleep(0.001)
        with _sp.region("io: save"):
            _time.sleep(0.003)
        profiling_results = _sp.finalize(return_results=True, verbose=False)

    profile_plots = OutputPlots(_FakeOutput(profiling_results)).profile
    save(PlotResult(*profile_plots.gantt()), "profile_gantt.png")
    save(PlotResult(*profile_plots.flame()), "profile_flame.png")
    save(PlotResult(*profile_plots.callgraph(compact=True)), "profile_callgraph.png")

    try:
        import plotly  # noqa: F401

        fig = profile_plots.gantt(backend="plotly")
        fig.write_json(PLOTLY_OUT / "plotly_profile_gantt.json")
        print(f"wrote {PLOTLY_OUT / 'plotly_profile_gantt.json'}")
    except ImportError:
        print("skipped plotly_profile_gantt.json (plotly unavailable)")
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped profile_*.png (scope-profiler unavailable): {exc}")

plt.close("all")
print("done")
