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
from struphy_plots.plotting import (PlotResult, plot_convergence,
                                    plot_dispersion, plot_scalars)

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
# Standalone interactive PyVista scenes (vtk.js inlined), loaded in an iframe by <PyVistaScene>.
PYVISTA_OUT = DOCS / "public" / "pyvista"
PYVISTA_OUT.mkdir(parents=True, exist_ok=True)


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
eta1 = np.linspace(0.0, 1.0, n_e1)
eta2 = np.linspace(0.0, 1.0, n_e2)
E1, E2 = np.meshgrid(eta1, eta2, indexing="ij")

radius = 0.28
center1 = 0.5 + radius * np.cos(t)
center2 = 0.5 + radius * np.sin(t)
sigma = 0.10

phi = np.empty((n_t, n_e1, n_e2))
for i in range(n_t):
    dist2 = (E1 - center1[i]) ** 2 + (E2 - center2[i]) ** 2
    phi[i] = np.exp(-dist2 / (2 * sigma**2)) * np.cos(10.0 * (E1 - center1[i]))

field = field_array(
    "phi", r"$\phi$", "a.u.", phi, ("t", "eta1", "eta2"), {"t": t, "eta1": eta1, "eta2": eta2}
)

save(field.struphy.plot.slice(x="eta1", y="eta2", t=-1), "slice.png")
save(field.struphy.plot.panels(x="eta1", y="eta2", nrows=2, ncols=3), "panels.png")
save(field.struphy.plot.lineout(x="eta1", t=-1, eta2=0.5), "lineout.png")

anim = field.struphy.plot.animation(x="eta1", y="eta2", step=2, interval=120)
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
    ("t", "component", "eta1", "eta2"),
    {"t": t, "eta1": eta1, "eta2": eta2},
)
save(
    vector.struphy.plot.vector(x="eta1", y="eta2", components=(0, 1), stride=6, t=-1),
    "vector.png",
)

# A 3-D scalar blob for the orthogonal-slices and volume renders
n3 = 40
e1_3, e2_3, e3_3 = (np.linspace(0.0, 1.0, n3) for _ in range(3))
E1_3, E2_3, E3_3 = np.meshgrid(e1_3, e2_3, e3_3, indexing="ij")
blob = np.exp(
    -((E1_3 - 0.5) ** 2 + (E2_3 - 0.5) ** 2 + (E3_3 - 0.5) ** 2) / (2 * 0.15**2)
)
volume_data = field_array(
    "n", "$n$", "a.u.", blob, ("eta1", "eta2", "eta3"), {"eta1": e1_3, "eta2": e2_3, "eta3": e3_3}
)
save(volume_data.struphy.plot.volume_slices(), "volume_slices.png")

try:
    import pyvista as pv

    pv.OFF_SCREEN = True

    physical = volume_data.assign_coords(
        X=(("eta1", "eta2", "eta3"), E1_3),
        Y=(("eta1", "eta2", "eta3"), E2_3),
        Z=(("eta1", "eta2", "eta3"), E3_3),
    )
    plotter = physical.struphy.plot.volume(cmap="viridis")
    plotter.camera_position = "iso"
    plotter.screenshot(str(OUT / "volume.png"))
    plotter.close()
    print(f"wrote {OUT / 'volume.png'}")
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(
        f"skipped volume.png (PyVista unavailable or headless rendering failed): {exc}"
    )


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
ax.plot(
    fit.time, fit.fitted, "--", color="C2", label=rf"fit: $\gamma$ = {fit.rate:.3f}"
)
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
decaying_field = field * xr.DataArray(
    1.0 / (1.0 + 0.4 * t), dims=("t",), coords={"t": t}
)
decaying_field.attrs = dict(field.attrs)
norm_t = decaying_field.struphy.analysis.norm()
save(
    norm_t.struphy.plot.lineout(x="t", title="Field norm decaying in time"), "norm.png"
)

save(
    total.struphy.analysis.drift().struphy.plot.lineout(
        x="t", title="Drift from the initial value"
    ),
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
    en_cons.struphy.analysis.relative_error().struphy.plot.lineout(
        x="t", title="Relative energy conservation error"
    ),
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

distribution = field_array(
    "f", "$f$", "a.u.", f, ("t", "eta1", "v1"), {"t": tp, "eta1": e1p, "v1": v1p}
)

save(distribution.struphy.plot.slice(x="eta1", y="v1", t=-1), "phase_space.png")

moments = distribution.struphy.analysis.velocity_moments()
final = moments.isel(t=-1)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), layout="constrained")
for ax, name in zip(axes, ("density", "mean_v1", "variance_v1")):
    array = final[name]
    ax.plot(array.eta1, array)
    ax.set(
        xlabel=axis_label(array, "eta1"),
        ylabel=value_label(array),
        title=array.attrs.get("label", name),
    )
save_fig(fig, "velocity_moments.png")

averaged = distribution.struphy.analysis.spatial_average()
save(
    averaged.struphy.plot.slice(
        x="t", y="v1", title="Velocity distribution averaged over space"
    ),
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
marker_orbits = orbits  # `orbits` becomes struphy_plots.theory.orbits further down

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
    ("eta1", "eta2"),
    {"eta1": e1_bg, "eta2": e2_bg},
)
n_orbit_steps, n_confined = 60, 6
s_orbit = np.linspace(0.0, 2 * np.pi, n_orbit_steps)
radii_orbit = np.linspace(0.15, 0.3, n_confined)
phases_orbit = rng_p.uniform(0, 2 * np.pi, n_confined)
orbit_e1 = 0.5 + radii_orbit[None, :] * np.cos(s_orbit[:, None] + phases_orbit[None, :])
orbit_e2 = 0.5 + radii_orbit[None, :] * np.sin(s_orbit[:, None] + phases_orbit[None, :])
confined_orbits = xr.Dataset(
    {"eta1": (("t", "marker"), orbit_e1), "eta2": (("t", "marker"), orbit_e2)},
    coords={"t": s_orbit, "marker": np.arange(n_confined)},
)
save(
    well.struphy.plot.overlay_orbits(confined_orbits, x="eta1", y="eta2"),
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
    wave += np.cos(k * X_DISP - omega_k * T_DISP) + np.cos(
        k * X_DISP + omega_k * T_DISP
    )
wave += 0.05 * rng_disp.standard_normal(wave.shape)

dispersive_field = field_array(
    "phi", r"$\phi$", "a.u.", wave, ("t", "eta1"), {"t": t_disp, "eta1": x_disp}
)
save(
    dispersive_field.struphy.plot.dispersion(
        branches={"Bohm-Gross": bohm_gross}, kmax=7, omega_max=12
    ),
    "dispersion.png",
)

# Broadband light waves, omega = |k|, both ways along z: a straight branch, fitted without theory
rng_light = np.random.default_rng(0)
t_light, z_light = np.arange(400) * 0.05, np.linspace(0.0, 20.0, 128, endpoint=False)
T_light, Z_light = np.meshgrid(t_light, z_light, indexing="ij")
light_values = np.zeros_like(T_light)
for n_light in range(1, 40):
    k_light = 2 * np.pi * n_light / 20
    for sign in (1, -1):
        light_values += rng_light.normal() * np.cos(k_light * Z_light + sign * k_light * T_light
                                                    + rng_light.uniform(0, 2 * np.pi))
e_x = xr.DataArray(light_values, dims=("t", "z"), coords={"t": t_light, "z": z_light}, name="E_x",
                   attrs={"label": "$E_x$"})
light_spectrum = e_x.struphy.analysis.dispersion(dim="z")
light_fits = light_spectrum.struphy.analysis.fit_branches(n_branches=1)
save(
    light_spectrum.struphy.plot.dispersion(kmin=0, branches={"light, ω = k": lambda k: k}, fits=light_fits,
                                           dynamic_range=12, omega_max=25),
    "dispersion_fits.png",
)


# =============================================================================
# A Plotly figure built by hand from .struphy.data (the Selecting data guide; needs
# `pip install plotly`): figure JSON for the docs' <PlotlyChart> component
# (docs/src/components/PlotlyChart.astro), which loads Plotly.js from a CDN. The
# backend="plotly" versions of the plots are written at the end of this script.
# =============================================================================
try:
    import plotly.express as px

    def save_plotly(fig, filename, *, width=560, height=440):
        # Only the JSON is used (fetched client-side by <PlotlyChart>, which sets its own
        # size); width/height here are just so the figure looks reasonable if opened directly.
        fig.update_layout(width=width, height=height)
        fig.write_json(PLOTLY_OUT / f"{filename}.json")
        print(f"wrote {PLOTLY_OUT / f'{filename}.json'}")

    selected = field.struphy.data.slice(x="eta1", y="eta2", t=-1)
    fig = px.imshow(
        selected.transpose("eta2", "eta1"),
        x=selected.eta1,
        y=selected.eta2,
        origin="lower",
        color_continuous_scale="viridis",
    )
    fig.update_layout(xaxis_title="eta1", yaxis_title="eta2")
    save_plotly(fig, "plotly_slice")
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped plotly_slice.json (plotly unavailable): {exc}")


# =============================================================================
# Spectral analysis: struphy_plots.spectral / spectral_plots
#
# A synthetic toroidal Alfven eigenmode (TAE) in a sixth of a hollow torus, with the
# parameters of Struphy's TAE tutorial: r = 0.1 + 0.9*eta1, q = 1.71 + 0.16 r^2,
# n0 = 1 - 0.8 r^2, B0 = 3, R0 = 10, harmonics m = 10, 11 with n = 6 (one per sector).
# Coupled m = 10/11 harmonics peaked either side of r* = 0.5 oscillate at the gap
# frequency and grow slowly; a continuum-damped m = 10 oscillation sits at r = 0.8.
# =============================================================================
from struphy_plots import spectral_plots as spp  # noqa: E402
from struphy_plots.spectral import (  # noqa: E402
    cross_spectrum,
    filter_time,
    matrix_pencil,
)

R0_tae, B0_tae, n_tae = 10.0, 3.0, 6


def q_tae(r):
    return 1.71 + 0.16 * np.asarray(r) ** 2


def vA_tae(r):
    return B0_tae / np.sqrt(1.0 - 0.8 * np.asarray(r) ** 2)


def alfven_continuum(r, m, n):
    """The uncoupled shear-Alfven continuum of harmonic (m, n), as for plot_continuous_spectrum."""
    return {"shear_Alfvén": vA_tae(r) / R0_tae * np.abs(n - m / q_tae(r))}


omega_tae = float(vA_tae(0.5) / (2 * q_tae(0.5) * R0_tae))
omega_continuum = float(alfven_continuum(0.8, 10, n_tae)["shear_Alfvén"])
s1, s2, s3 = np.linspace(0, 1, 28), np.linspace(0, 1, 49), np.linspace(0, 1, 9)
S1, S2, S3 = np.meshgrid(s1, s2, s3, indexing="ij")
r_tae = 0.1 + 0.9 * S1
t_tae = np.arange(300) * 2.0
rng_tae = np.random.default_rng(7)
tae_snapshots = []
for ti in t_tae:
    helical10 = 2 * np.pi * (10 * S2 - S3)
    helical11 = 2 * np.pi * (11 * S2 - S3)
    snapshot = np.exp(0.003 * ti) * (
        np.exp(-(((r_tae - 0.45) / 0.1) ** 2)) * np.cos(helical10 - omega_tae * ti)
        + 0.8 * np.exp(-(((r_tae - 0.56) / 0.1) ** 2)) * np.cos(helical11 - omega_tae * ti + 0.3)
    ) + 0.6 * np.exp(-0.004 * ti) * np.exp(-(((r_tae - 0.8) / 0.04) ** 2)) * np.cos(
        helical10 - omega_continuum * ti
    )
    tae_snapshots.append(snapshot + 0.02 * rng_tae.standard_normal(snapshot.shape))
phi_tae = field_array(
    "phi",
    r"$\phi$",
    "a.u.",
    np.stack(tae_snapshots),
    ("t", "eta1", "eta2", "eta3"),
    {"t": t_tae, "eta1": s1, "eta2": s2, "eta3": s3},
)
radius_of = lambda eta1: 0.1 + 0.9 * eta1  # noqa: E731

band_tae = phi_tae.struphy.analysis.filter_time(dims=("eta1", "eta2", "eta3"), pad_bins=3)
save(
    phi_tae.struphy.plot.power_spectrum(
        peaks=2, band=band_tae, frequencies={"TAE gap-centre estimate": omega_tae}, omega_max=0.5,
        title="Power spectrum, averaged over the torus sector",
    ),
    "spectral_power.png",
)
save(phi_tae.struphy.plot.filtered(band_tae, eta1=0.4, eta2=0.0, eta3=0.0), "spectral_filtered.png")
save(phi_tae.struphy.plot.mode_amplitudes(top=2, fit=(100.0, 500.0)), "spectral_mode_amplitudes.png")
save(phi_tae.struphy.plot.mode_map(t=-1, m_range=(0, 16), n_range=(-3, 3)), "spectral_mode_map.png")
save(
    phi_tae.struphy.plot.radial_power(
        x_of=radius_of, xlabel=r"$r/a$", continuum=(alfven_continuum, [(10, n_tae), (11, n_tae)]),
        omega_max=0.3,
    ),
    "spectral_radial_power.png",
)
peaks_tae = phi_tae.struphy.analysis.spectral_peaks(n_peaks=2, window="hann")
omega_measured = float(peaks_tae.omega_refined[0])
save(
    phi_tae.struphy.plot.mode_profiles(omega_measured, x_of=radius_of, xlabel=r"$r/a$", top=2),
    "spectral_mode_profiles.png",
)

# The TAE period is ~66: a run to t = 20 covers a third of it, below the first FFT bin.
t_short = np.arange(21) * 1.0
probe_short = field_array(
    "phi",
    r"$\phi$ at $r_*$",
    "a.u.",
    np.exp(0.003 * t_short) * np.cos(omega_tae * t_short + 0.4) + 1e-3 * rng_tae.standard_normal(21),
    ("t",),
    {"t": t_short},
)
save(probe_short.struphy.plot.pencil_fit(n_modes=1), "spectral_pencil.png")

# An energetic-particle-like mode chirping down in frequency, beside a steady mode
t_chirp = np.arange(1600) * 0.5
omega_chirp = 0.3 - 0.15 * t_chirp / t_chirp[-1]
chirp = field_array(
    "phi",
    r"$\phi$",
    "a.u.",
    np.sin(np.cumsum(omega_chirp) * 0.5) + 0.5 * np.cos(0.1 * t_chirp),
    ("t",),
    {"t": t_chirp},
)
save(chirp.struphy.plot.spectrogram(length=200.0, step=10.0, omega_max=0.45), "spectral_spectrogram.png")

# A standing shear-Alfven wave: velocity and magnetic perturbation a quarter period apart
z_wave = np.linspace(0.02, np.pi / 2 - 0.02, 24)
t_wave = np.arange(512) * 0.25
omega_wave = 2 * np.pi * 16 / 128
noise = rng_tae.normal(0, 0.15, (2, 512, 24))
u_wave = field_array(
    "u", "$u_x$", "a.u.", np.cos(z_wave)[None] * np.sin(omega_wave * t_wave)[:, None] + noise[0],
    ("t", "eta3"), {"t": t_wave, "eta3": z_wave},
)
b_wave = field_array(
    "b", "$b_x$", "a.u.", np.sin(z_wave)[None] * np.cos(omega_wave * t_wave)[:, None] + noise[1],
    ("t", "eta3"), {"t": t_wave, "eta3": z_wave},
)
save(u_wave.struphy.plot.cross_spectrum(b_wave, dims="eta3", omega_max=1.5), "spectral_cross.png")


# =============================================================================
# Run-level tools from struphy's TAE_example_Shrut branch: energy budgets,
# profiles at several times, mode profiles at one time, orbit projections
# =============================================================================
from struphy_plots.plotting import plot_energy_budget  # noqa: E402

# An energetic-particle drive: the wave grows while the energetic ions lose the same energy
t_en = np.linspace(0.0, 60.0, 301)
wave = 1e-3 * np.exp(0.08 * t_en) / (1 + 1e-3 * (np.exp(0.08 * t_en) - 1) / 0.2)
wiggle = 0.5 + 0.5 * np.cos(2 * 0.3 * t_en)
budget = xr.Dataset(
    {
        "en_U": ("t", wave * wiggle),
        "en_B": ("t", wave * (1 - wiggle) * 0.9),
        "en_p": ("t", wave * (1 - wiggle) * 0.1),
        "en_fv": ("t", 1.0 - 0.8 * (wave - wave[0])),
        "en_fB": ("t", 0.5 - 0.2 * (wave - wave[0])),
    },
    coords={"t": t_en},
)
budget["en_tot"] = sum(budget[name] for name in ("en_U", "en_B", "en_p", "en_fv", "en_fB")) * (1 + 2e-6 * t_en / 60)
save(
    plot_energy_budget(budget, groups={"wave": ["en_U", "en_B", "en_p"], "energetic ions": ["en_fv", "en_fB"]}),
    "energy_budget.png",
)

# Radial profiles of the synthetic TAE at a few times, and its harmonics at the last time
save(
    phi_tae.struphy.plot.profiles(x="eta1", at=[0, 100, 200, 299], x_of=radius_of, xlabel=r"$r/a$", eta2=0.0, eta3=0.0),
    "profiles.png",
)
save(
    phi_tae.struphy.plot.mode_profiles(t=-1, scale=(1, 6), x_of=radius_of, xlabel=r"$r/a$", top=2),
    "mode_profiles_snapshot.png",
)

# Guiding-centre-like orbits in a torus: passing markers circle the axis, trapped ones bounce
# Initial (v_par, mu) from a Maxwellian; markers inside the cone v_par^2 < 2 mu dB/B bounce
rng_orb = np.random.default_rng(11)
n_gc, t_gc = 60, np.linspace(0, 80, 400)[:, None]
r_gc, th_gc = rng_orb.uniform(0.2, 0.8, n_gc), rng_orb.uniform(0, 2 * np.pi, n_gc)
v0_gc, mu0_gc = rng_orb.normal(0.0, 1.0, n_gc), rng_orb.exponential(0.5, n_gc)
trapped_gc = v0_gc**2 < 2 * mu0_gc * 0.6
bounce_gc = 0.2 * t_gc + th_gc
v_par_gc = np.where(trapped_gc, v0_gc * np.cos(0.2 * t_gc), v0_gc * (1 - 0.3 * np.sin(0.2 * t_gc) ** 2))
theta_gc = th_gc + np.where(
    trapped_gc, 1.2 * np.sin(bounce_gc) - 1.2 * np.sin(th_gc), 0.25 * np.sign(v0_gc) * t_gc
)
R_gc = 3.0 + (r_gc + 0.06 * np.where(trapped_gc, np.sin(bounce_gc), 0)) * np.cos(theta_gc)
phi_gc = 0.1 * t_gc
x_gc, y_gc = R_gc * np.cos(phi_gc), R_gc * np.sin(phi_gc)
z_gc = (r_gc + 0.06 * np.where(trapped_gc, np.sin(bounce_gc), 0)) * np.sin(theta_gc)
mu_gc = np.broadcast_to(mu0_gc, x_gc.shape) * (1 + 1e-7 * np.sin(t_gc))
for arr in (x_gc, y_gc, z_gc, v_par_gc):
    arr[300:, 1] = 0.0  # one marker leaves the domain
mu_gc = mu_gc.copy()
mu_gc[300:, 1] = 0.0
orbits_gc = xr.Dataset(
    {
        "x": (("t", "marker"), x_gc),
        "y": (("t", "marker"), y_gc),
        "z": (("t", "marker"), z_gc),
        "v_par": (("t", "marker"), v_par_gc, {"label": r"$v_\parallel$"}),
        "mu": (("t", "marker"), mu_gc, {"label": r"$\mu$"}),
    },
    coords={"t": t_gc[:, 0], "marker": np.arange(n_gc)},
)
e1_b, e2_b = np.linspace(0, 1, 5), (np.arange(64) + 0.5) / 64
E1_b, E2_b = np.meshgrid(e1_b, e2_b, indexing="ij")
boundary_field = field_array(
    "b", "b", "", np.ones((5, 64, 1)), ("eta1", "eta2", "eta3"), {"eta1": e1_b, "eta2": e2_b, "eta3": [0.0]}
).assign_coords(
    X=(("eta1", "eta2", "eta3"), ((3.0 + (0.1 + 0.9 * E1_b) * np.cos(2 * np.pi * E2_b)))[..., None]),
    Y=(("eta1", "eta2", "eta3"), np.zeros((5, 64, 1))),
    Z=(("eta1", "eta2", "eta3"), ((0.1 + 0.9 * E1_b) * np.sin(2 * np.pi * E2_b))[..., None]),
)
save(orbits_gc.struphy.plot.orbit_classification(), "orbit_classification.png")
save(orbits_gc.struphy.plot.poloidal(boundary=boundary_field), "orbits_poloidal.png")

# The shear-Alfven continua of the TAE harmonics, with the frequency measured above
from struphy_plots.plotting import plot_continuous_spectrum  # noqa: E402

save(
    plot_continuous_spectrum(
        alfven_continuum,
        np.linspace(0.1, 1.0, 300),
        [(10, n_tae), (11, n_tae)],
        frequencies={"measured TAE frequency": omega_measured},
        xlabel=r"$r/a$",
        title="Shear-Alfvén continua of the m = 10, 11 harmonics",
    ),
    "continuous_spectrum.png",
)
save(orbits_gc.struphy.plot.quantities(markers=4), "orbits_quantities.png")


# =============================================================================
# Tools for the struphy-hub examples: contour lines, markers over fields and in
# motion, marker paths, side-by-side animations, gradients, relative modes
# =============================================================================
def unit_square(values, n1, n2, *, t=None, name="f", label="f", scale=(1.0, 1.0), offset=(0.0, 0.0)):
    """A (t,) eta1, eta2, flat eta3 field on a rectangle x = offset + scale * eta."""
    e1, e2 = (np.arange(n1) + 0.5) / n1, (np.arange(n2) + 0.5) / n2
    E1, E2 = np.meshgrid(e1, e2, indexing="ij")
    X, Y = offset[0] + scale[0] * E1, offset[1] + scale[1] * E2
    dims = ("eta1", "eta2", "eta3") if t is None else ("t", "eta1", "eta2", "eta3")
    coords = {"eta1": e1, "eta2": e2, "eta3": [0.0]}
    if t is not None:
        coords["t"] = t
    coords.update({n: (("eta1", "eta2", "eta3"), c[..., None]) for n, c in zip("XYZ", (X, Y, 0 * X))})
    return field_array(name, label, "a.u.", values[..., None], dims, coords), X, Y


# Diocotron-like: a charged ring in an annulus, whose m = 4 interface wave grows
ring_t = np.linspace(0.0, 20.0, 41)
a1, a2 = 48, 96
r1, th1 = (np.arange(a1) + 0.5) / a1, (np.arange(a2) + 0.5) / a2
R1, TH1 = np.meshgrid(0.1 + 0.9 * r1, 2 * np.pi * th1, indexing="ij")
ring_values = np.stack(
    [
        np.exp(-(((R1 - 0.55 - 0.05 * np.exp(0.25 * (ti - 20)) * np.cos(4 * TH1 - 0.3 * ti)
                   - 0.01 * np.exp(0.1 * (ti - 20)) * np.cos(3 * TH1)) / 0.08) ** 4))
        for ti in ring_t
    ]
)
e1r, e2r = r1, th1
ring = field_array(
    "n", "$n$", "a.u.", ring_values[..., None], ("t", "eta1", "eta2", "eta3"),
    {"t": ring_t, "eta1": e1r, "eta2": e2r, "eta3": [0.0]},
).assign_coords(
    X=(("eta1", "eta2", "eta3"), (R1 * np.cos(TH1))[..., None]),
    Y=(("eta1", "eta2", "eta3"), (R1 * np.sin(TH1))[..., None]),
    Z=(("eta1", "eta2", "eta3"), np.zeros((a1, a2, 1))),
)
save(
    ring.struphy.plot.slice(coords="physical", plane="XY", t=-1, eta3=0, levels=[0.2], cmap="viridis",
                            title="Charge density with the interface n = 0.2"),
    "contour_interface.png",
)
save(
    # at the outer interface, where the wave moves the edge (a radial average would cancel it)
    ring.isel(eta3=0).sel(eta1=0.59, method="nearest").struphy.plot.mode_amplitudes(
        dims="eta2", names="m", relative=True, top=3, fit=(8.0, 20.0)
    ),
    "relative_modes.png",
)

# Dam-break-like: a column of fluid markers collapses to the right; its density follows
rng_db = np.random.default_rng(5)
n_db = 900
x0_db, y0_db = rng_db.uniform(0.0, 0.3, n_db), rng_db.uniform(0.0, 0.6, n_db)
t_db = np.linspace(0.0, 3.0, 31)
spread = 1 / (1 + np.exp(-3 * (t_db - 1.5)))  # 0 -> 1
x_db = x0_db[None] + (x0_db[None] / 0.3 * 0.65 + 0.05) * spread[:, None]
y_db = y0_db[None] * (1 - 0.72 * spread[:, None]) + 0.02 * np.sin(8 * x_db) * spread[:, None]
dam = xr.Dataset(
    {"x": (("t", "marker"), x_db), "y": (("t", "marker"), y_db)},
    coords={"t": t_db, "marker": np.arange(n_db)},
    attrs={"label": "fluid markers"},
)
g1, g2 = (np.arange(40) + 0.5) / 40, (np.arange(40) + 0.5) / 40
G1, G2 = np.meshgrid(g1, g2, indexing="ij")
kernel_width = 0.04
dam_density = np.stack(
    [
        np.exp(-((G1[..., None] - x_db[i]) ** 2 + (G2[..., None] - y_db[i]) ** 2) / (2 * kernel_width**2)).sum(-1)
        / (n_db * 2 * np.pi * kernel_width**2)
        for i in range(len(t_db))
    ]
)
dam_field, _, _ = unit_square(dam_density, 40, 40, t=t_db, name="n", label="$n$")
save(
    dam.struphy.plot.scatter(x="x", y="y", color="x", color_at=0, t=-1, s=5, cmap="plasma",
                             background=dam_field.isel(eta3=0), background_options={"cmap": "Blues"}),
    "markers_over_density.png",
)
animation_db = dam.struphy.plot.animation(
    x="x", y="y", color="x", color_at=0, s=5, cmap="plasma",
    background=dam_field.isel(eta3=0), background_options={"cmap": "Blues"},
)
animation_db.save(PUBLIC_OUT / "markers_animation.gif", writer="pillow", fps=8)
print(f"wrote {PUBLIC_OUT / 'markers_animation.gif'}")

# Beltrami-like: markers circulating along the streamlines of psi = sin(2 pi x) sin(2 pi y)
n_bs = 64
psi, XB, YB = unit_square(
    np.zeros((n_bs, n_bs)), n_bs, n_bs, name="psi", label=r"$\psi$", offset=(-0.5, -0.5)
)
psi = psi.copy(data=(np.sin(2 * np.pi * XB) * np.sin(2 * np.pi * YB))[..., None] / (2 * np.pi))


def cellular_velocity(p):
    x_, y_ = p[..., 0], p[..., 1]
    return np.stack([np.sin(2 * np.pi * x_) * np.cos(2 * np.pi * y_), -np.cos(2 * np.pi * x_) * np.sin(2 * np.pi * y_)], axis=-1)


starts = np.column_stack([np.linspace(0.28, 0.46, 6), np.full(6, 0.25)])  # from a cell centre outwards
starts = np.vstack([starts, rng_db.uniform(-0.45, 0.45, (26, 2))])
t_bs, dt_bs = np.linspace(0, 3, 301), 0.01
path = [starts]
for _ in range(len(t_bs) - 1):  # RK4
    p0 = path[-1]
    k1 = cellular_velocity(p0)
    k2 = cellular_velocity(p0 + 0.5 * dt_bs * k1)
    k3 = cellular_velocity(p0 + 0.5 * dt_bs * k2)
    k4 = cellular_velocity(p0 + dt_bs * k3)
    path.append(p0 + dt_bs * (k1 + 2 * k2 + 2 * k3 + k4) / 6)
path = np.array(path)
beltrami = xr.Dataset(
    {"x": (("t", "marker"), path[..., 0]), "y": (("t", "marker"), path[..., 1])},
    coords={"t": t_bs, "marker": np.arange(path.shape[1])},
)
save(
    beltrami.struphy.plot.paths(
        near=[tuple(p) for p in starts[:6]],
        background=psi.isel(eta3=0),
        background_options={"levels": 12, "fill": False, "cmap": "Greys", "title": "Marker paths over the streamlines"},
    ),
    "marker_paths.png",
)

# Hasegawa-Wakatani-like: drift-wave eddies that give way to a zonal flow
L_hw, n_hw = 20.0, 64
t_hw = np.linspace(0.0, 30.0, 31)
hw_e = (np.arange(n_hw) + 0.5) / n_hw
HX, HY = np.meshgrid(L_hw * hw_e, L_hw * hw_e, indexing="ij")
kx0, ky0 = 2 * np.pi / L_hw, 2 * np.pi / L_hw
rng_hw = np.random.default_rng(2)
modes_hw = [(rng_hw.integers(1, 5), rng_hw.integers(1, 5), rng_hw.uniform(0, 2 * np.pi), rng_hw.uniform(0.3, 1.0)) for _ in range(12)]
phi_values, omega_values, n_values = [], [], []
for ti in t_hw:
    zonal_amp, wave_amp = 0.6 * np.tanh(ti / 12), np.exp(-ti / 25)
    phi_t = zonal_amp * np.sin(kx0 * 2 * HX)
    omega_t = -zonal_amp * (2 * kx0) ** 2 * np.sin(kx0 * 2 * HX)
    n_t = 0.3 * zonal_amp * np.sin(kx0 * 2 * HX)
    for mx, my, phase, amp in modes_hw:
        wave = np.cos(mx * kx0 * HX + my * ky0 * HY - 0.4 * my * ti + phase)
        phi_t = phi_t + wave_amp * amp * 0.3 * wave
        omega_t = omega_t - wave_amp * amp * 0.3 * ((mx * kx0) ** 2 + (my * ky0) ** 2) * wave
        n_t = n_t + wave_amp * amp * 0.3 * np.cos(mx * kx0 * HX + my * ky0 * HY - 0.4 * my * ti + phase + 0.5)
    phi_values.append(phi_t)
    omega_values.append(omega_t)
    n_values.append(n_t)
phi_hw, _, _ = unit_square(np.array(phi_values), n_hw, n_hw, t=t_hw, name="phi", label=r"$\phi$", scale=(L_hw, L_hw))
omega_hw = phi_hw.copy(data=np.array(omega_values)[..., None]).rename("omega")
omega_hw.attrs.update(label=r"$\omega$")
density_hw = phi_hw.copy(data=np.array(n_values)[..., None]).rename("n")
density_hw.attrs.update(label="$n$")
animation_hw = omega_hw.struphy.plot.animation(
    coords="physical", plane="XY", eta3=0, alongside=[density_hw], cmap="RdBu_r", symmetric=True, robust=True
)
animation_hw.save(PUBLIC_OUT / "fields_side_by_side.gif", writer="pillow", fps=6)
print(f"wrote {PUBLIC_OUT / 'fields_side_by_side.gif'}")

# Recipe: the E x B kinetic energy of drift waves and of the zonal flow, from gradient()
grad_phi = phi_hw.struphy.analysis.gradient()
total_energy = 0.5 * (grad_phi.sel(component=[0, 1]) ** 2).sum("component").mean(("eta1", "eta2", "eta3"))
zonal = xr.zeros_like(phi_hw) + phi_hw.mean("eta2")  # keeps the X, Y coordinates
grad_zonal = zonal.struphy.analysis.gradient()
zonal_energy = 0.5 * (grad_zonal.sel(component=[0, 1]) ** 2).sum("component").mean(("eta1", "eta2", "eta3"))
fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.stackplot(
    t_hw, (total_energy - zonal_energy).values, zonal_energy.values,
    labels=[r"drift waves ($k_y \neq 0$)", r"zonal flow ($k_y = 0$)"], colors=["#168aad", "#f08a4b"], alpha=0.85,
)
ax.set(xlabel="$t$", ylabel=r"$\frac{1}{2}\langle|\nabla\phi|^2\rangle$", title="E×B kinetic energy")
ax.legend(loc="upper right", fontsize="small")
save_fig(fig, "zonal_energy.png")


# =============================================================================
# Comparing with theory (from the survey of all struphy-hub examples): exact
# curves over time series and profiles, errors, mode projections, traced
# dispersion branches, overlays on slices, orbit grids
# =============================================================================
from struphy_plots.plotting import plot_measured_vs_theory  # noqa: E402

# A damped standing Langmuir wave, as in the Landau-damping examples; project_mode picks its
# amplitude out of the field, and the exact envelope goes over it
gamma_ld, omega_ld, k_ld = 0.15, 1.4, 1
x_ld = (np.arange(64) + 0.5) / 64
t_ld = np.linspace(0.0, 25.0, 251)
rng_ld = np.random.default_rng(11)
e_ld = field_array(
    "e1", "$E_x$", "a.u.",
    0.1 * np.exp(-gamma_ld * t_ld)[:, None] * np.cos(omega_ld * t_ld)[:, None] * np.sin(2 * np.pi * k_ld * x_ld)[None]
    + 2e-3 * rng_ld.standard_normal((251, 64)),
    ("t", "eta1"), {"t": t_ld, "eta1": x_ld},
)
amplitude_ld = e_ld.struphy.analysis.project_mode(dim="eta1", number=k_ld)
amplitude_ld.attrs.update(label=r"$\hat E_{k=1}$")
save(
    amplitude_ld.struphy.plot.timeseries(
        logy=False,
        reference={"$\\pm 0.1\\,e^{-\\gamma t}$": lambda t: 0.1 * np.exp(-gamma_ld * t),
                   "_lower": lambda t: -0.1 * np.exp(-gamma_ld * t)},
        title="Mode k = 1 of the field against the exact envelope",
    ),
    "reference_timeseries.png",
)

# A diffusing Gaussian whose numerical diffusion coefficient is 10 % too large
D_exact, D_run = 0.01, 0.011
x_hd = np.linspace(0.0, 1.0, 101)
t_hd = np.linspace(0.0, 2.0, 41)


def heat_kernel(x, t, D=D_exact, width=0.05):
    s2 = width**2 + 2 * D * t
    return width / np.sqrt(s2) * np.exp(-((x - 0.5) ** 2) / (2 * s2))


heat = field_array("T", "$T$", "a.u.", heat_kernel(x_hd[None], t_hd[:, None], D_run), ("t", "eta1"), {"t": t_hd, "eta1": x_hd})
save(
    heat.struphy.plot.profiles(x="eta1", at=[0, 10, 20, 40], reference={"exact": heat_kernel},
                               title="Temperature profiles and the exact solution"),
    "profiles_exact.png",
)
animation_hd = heat.struphy.plot.line_animation(reference={"exact": heat_kernel}, step=2)
animation_hd.save(PUBLIC_OUT / "line_animation.gif", writer="pillow", fps=8)
print(f"wrote {PUBLIC_OUT / 'line_animation.gif'}")
errors_hd = xr.Dataset({
    "rms": heat.struphy.analysis.error(heat_kernel, relative=True),
    "max": heat.struphy.analysis.error(heat_kernel, norm="max", relative=True),
})
fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
for name, series in errors_hd.items():
    ax.plot(t_hd, series, label=f"relative {name} error")
ax.set(xlabel="$t$", ylabel="error", title="Error against the exact solution")
ax.legend()
save_fig(fig, "error_in_time.png")

# The Bohm-Gross branch, traced in the dispersion diagram and compared with the theory
spectrum_bg = struphy_plots.analysis.power_spectrum(dispersive_field)
traced_bg = spectrum_bg.struphy.analysis.trace_branch(bohm_gross, window=0.2, k_range=(1.5, 5.5)).dropna("k")
save(
    dispersive_field.struphy.plot.dispersion(
        branches={"Bohm-Gross": bohm_gross}, frequencies={"plasma frequency": 1.0},
        points={"traced": traced_bg}, kmax=7, omega_max=12,
    ),
    "dispersion_traced.png",
)
save(
    plot_measured_vs_theory(traced_bg.omega, {"Bohm-Gross": bohm_gross, "cold plasma": lambda k: 1.0 + 0 * k},
                            xlabel="$k$", ylabel=r"$\omega$", title="Traced frequencies against theory"),
    "measured_vs_theory.png",
)

# Magnetic islands: |B| with the flux function's contours, its O- and X-points and the grid
n_is = 96
b_is, XI, YI = unit_square(np.zeros((n_is, n_is)), n_is, n_is, scale=(2 * np.pi, 2 * np.pi))
flux_exact = -np.cos(YI) - 0.3 * np.cos(XI)
b_is = xr.DataArray(
    np.stack([np.sin(YI), -0.3 * np.sin(XI), 0 * XI])[..., None],  # B = (dA/dy, -dA/dx, 0)
    dims=("component", "eta1", "eta2", "eta3"),
    coords={"component": [0, 1, 2], **b_is.coords}, name="B", attrs={"label": "B"},
)
flux_is = b_is.struphy.analysis.flux_function()
strength = np.sqrt((b_is**2).sum("component")).rename("absB")
strength.attrs.update(label="$|B|$")
save(
    strength.struphy.plot.slice(
        coords="physical", plane="XY", eta3=0, cmap="magma",
        overlays={"contours_of": flux_is, "contour_levels": 14, "contour_color": "w", "boundary": True,
                  "points": {"O-point": (np.pi, np.pi), "X-points": ([0.05, 2 * np.pi - 0.05], [np.pi, np.pi])}},
        title="|B| with flux surfaces from flux_function()",
    ),
    "overlay_flux.png",
)

# A wave packet on a space-time map, with the line it should follow
x_wp = (np.arange(128) + 0.5) / 128
t_wp = np.linspace(0.0, 1.5, 61)
packet = field_array(
    "phi", r"$\phi$", "a.u.",
    np.exp(-((x_wp[None] - 0.2 - 0.4 * t_wp[:, None]) ** 2) / 0.003)
    * np.cos(60 * (x_wp[None] - 0.5 * t_wp[:, None])),
    ("t", "eta1"), {"t": t_wp, "eta1": x_wp},
)
save(
    packet.struphy.plot.slice(
        x="eta1", y="t", cmap="RdBu_r", symmetric=True,
        overlays={"lines": {"group velocity 0.4": lambda x: (x - 0.2) / 0.4}, "line_color": "k"},
        title="Wave packet moving at the group velocity",
    ),
    "overlay_spacetime.png",
)

# Individual guiding-center orbits side by side, with their invariants and bounce periods
save(orbits_gc.struphy.plot.orbit_grid(markers=8, ncols=4, boundary=boundary_field), "orbit_grid.png")

# Selecting data: the ring's last slice and a radial cut from .struphy.data, drawn with plain
# matplotlib, with the densest point and the ring's extent found on the selected arrays
last = ring.struphy.data.slice(coords="physical", plane="XY", t=-1, eta3=0)
cut = ring.struphy.data.lineout(x="eta1", t=-1, eta2=0.3, eta3=0)
radius = 0.1 + 0.9 * cut.eta1
densest = last.isel(last.argmax(...))
fig, (ax_map, ax_cut) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained", width_ratios=(1, 1.2))
filled = ax_map.contourf(last.X, last.Y, last, levels=12, cmap="viridis")
ax_map.plot(float(densest.X), float(densest.Y), "w*", ms=12, label="maximum")
angle = 2 * np.pi * float(cut.eta2)
ax_map.plot(radius * np.cos(angle), radius * np.sin(angle), "w--", lw=1.2, label="cut")
ax_map.set(aspect="equal", xlabel="X", ylabel="Y", title="data.slice(...) drawn with contourf")
ax_map.legend(loc="lower left", fontsize="small", facecolor="0.25", labelcolor="w")
fig.colorbar(filled, ax=ax_map, label="$n$")
ax_cut.plot(radius, cut, color="#168aad")
inside = radius.where(cut >= 0.2)
ax_cut.axvspan(float(inside.min()), float(inside.max()), color="#f08a4b", alpha=0.25, label="n ≥ 0.2")
ax_cut.set(xlabel="r", ylabel="$n$", title="data.lineout(...) along the cut")
ax_cut.legend(fontsize="small")
save_fig(fig, "data_selection.png")


# =============================================================================
# Theory: struphy_plots.theory, the analytic results to compare runs against
# =============================================================================
from struphy_plots.theory import exact, kinetic, numerics, orbits, waves  # noqa: E402

# Landau damping of Langmuir waves: the kinetic root against Bohm-Gross and the weak-damping formula
k_th = np.linspace(0.1, 0.6, 101)
root = kinetic.langmuir(k_th)
weak = kinetic.landau_damping_weak(k_th)
fig, (ax_w, ax_g) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
ax_w.plot(k_th, root.real, label="kinetic root, langmuir()")
ax_w.plot(k_th, kinetic.bohm_gross(k_th).real, "--", label="Bohm–Gross")
ax_w.set(xlabel=r"$k\lambda_D$", ylabel=r"$\omega/\omega_p$", title="Frequency")
ax_g.semilogy(k_th, -root.imag, label="kinetic root")
ax_g.semilogy(k_th, -weak.imag, "--", label="weak-damping formula")
ax_g.set(xlabel=r"$k\lambda_D$", ylabel=r"$-\gamma/\omega_p$", title="Landau damping rate", ylim=(1e-8, 1))
for ax in (ax_w, ax_g):
    ax.legend(fontsize="small")
save_fig(fig, "theory_landau.png")

# The workflow: theory functions as branches= of a measured dispersion diagram (a synthetic field
# of damped Langmuir waves at a few wavenumbers)
L_th, n_th = 20 * np.pi, 256
x_th = np.linspace(0, L_th, n_th, endpoint=False)
t_th = np.linspace(0, 80, 800)
langmuir_field = np.zeros((t_th.size, n_th))
for mode in range(1, 11):  # k = 0.1 ... 1
    k_mode = 2 * np.pi * mode / L_th
    w_mode = kinetic.langmuir(k_mode)
    langmuir_field += np.exp(w_mode.imag * t_th)[:, None] * np.cos(k_mode * x_th[None] - w_mode.real * t_th[:, None])
e_th = field_array("e1", "$E_x$", "a.u.", langmuir_field, ("t", "eta1"), {"t": t_th, "eta1": x_th})
save(
    e_th.struphy.plot.dispersion(
        branches={"kinetic": kinetic.langmuir, "Bohm–Gross": kinetic.bohm_gross}, kmax=1.1, omega_max=2.5,
        title="Theory as branches=: Langmuir waves",
    ),
    "theory_dispersion.png",
)

# Instabilities: bump-on-tail and two-stream growth rates, and the Weibel instability
k_bot = np.linspace(0.05, 0.5, 120)
k_ts = np.linspace(0.05, 1.2, 120)
fig, (ax_e, ax_m) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
ax_e.plot(k_bot, kinetic.bump_on_tail(k_bot).imag, label="bump-on-tail ($n_b$ = 0.1, $v_b$ = 4.5)")
ax_e.plot(k_ts, kinetic.two_stream(k_ts, beam_speed=1.0, thermal_speed=0.1).imag, label="two-stream, warm")
ax_e.plot(k_ts, kinetic.two_stream_cold(k_ts, beam_speed=1.0).imag, "--", label="two-stream, cold")
ax_e.axhline(0, color="k", lw=0.8)
ax_e.set(xlabel=r"$k\lambda_D$", ylabel=r"$\gamma/\omega_p$", title="Electrostatic instabilities", ylim=(-0.05, 0.4))
k_wb = np.linspace(0.05, 2.2, 120)
for anisotropy in (2.0, 4.0, 6.0):
    ax_m.plot(k_wb, kinetic.weibel(k_wb, anisotropy=anisotropy, parallel_thermal_speed=0.1).imag,
              label=rf"$T_\perp/T_\parallel$ = {anisotropy:g}")
ax_m.axhline(0, color="k", lw=0.8)
ax_m.set(xlabel=r"$kc/\omega_p$", ylabel=r"$\gamma/\omega_p$", title="Weibel instability", ylim=(-0.02, None))
for ax in (ax_e, ax_m):
    ax.legend(fontsize="small")
save_fig(fig, "theory_instabilities.png")

# MHD and cold-plasma waves: the Friedrichs diagram, and cold-plasma branches at 45°
theta_f = np.linspace(0, 2 * np.pi, 361)
speeds = waves.magnetosonic_speeds(theta_f, alfven_speed=1.0, sound_speed=0.6)
fig = plt.figure(figsize=(10, 4), layout="constrained")
ax_f = fig.add_subplot(1, 2, 1, projection="polar")
for name, style in (("fast", "-"), ("shear Alfvén", "--"), ("slow", ":")):
    ax_f.plot(theta_f, np.abs(speeds[name]), style, label=name)
ax_f.set_title(r"Phase speeds, $c_s/v_A$ = 0.6 ($\mathbf{B}$ along 0°)")
ax_f.legend(fontsize="small", loc="lower left", bbox_to_anchor=(-0.15, -0.12))
ax_c = fig.add_subplot(1, 2, 2)
plasma = waves.electron_ion(plasma_frequency=1.0, cyclotron_frequency=0.6, mass_ratio=25)
k_cp = np.linspace(0.01, 4, 300)
for name, omega in waves.cold_plasma_waves(k_cp, np.pi / 4, plasma).items():
    ax_c.plot(k_cp, omega.real, label=name)
ax_c.plot(k_cp, k_cp, "k:", lw=0.8, label=r"$\omega = ck$")
ax_c.set(xlabel=r"$kc/\omega_{pe}$", ylabel=r"$\omega/\omega_{pe}$", ylim=(0, 3),
         title=r"Cold plasma at 45°, $m_i/m_e$ = 25")
ax_c.legend(fontsize="x-small", ncol=2)
save_fig(fig, "theory_waves.png")

# Hall MHD along B, and the toroidal Alfvén continuum with its TAE gap
fig, (ax_h, ax_t) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
k_h = np.linspace(0.01, 4, 200)
for name, omega in waves.hall_mhd_parallel(k_h, alfven_speed=1.0, ion_inertial_length=1.0).items():
    ax_h.plot(k_h, omega.real, label=name)
ax_h.plot(k_h, k_h, "k:", lw=0.8, label=r"ideal MHD, $\omega = kv_A$")
ax_h.set(xlabel=r"$k d_i$", ylabel=r"$\omega/\Omega_i$", ylim=(0, 6), title="Hall MHD along B")
ax_h.legend(fontsize="small")
r_t = np.linspace(0.05, 1.0, 400)
R0_t = 5.0
q_t = lambda r: 1.0 + r**2  # noqa: E731
for m_t in (1, 2):
    ax_t.plot(r_t, waves.alfven_continuum(r_t, m=m_t, n=-1, q=q_t, major_radius=R0_t).real * R0_t, label=f"m = {m_t}")
ax_t.axhline(waves.tae_frequency(1.5, major_radius=R0_t) * R0_t, color="k", lw=0.8, ls="--",
             label="TAE frequency, q = 1.5")
ax_t.set(xlabel="r/a", ylabel=r"$\omega R_0/v_A$", ylim=(0, 0.8), title="Alfvén continua, n = −1, q = 1 + r²")
ax_t.legend(fontsize="x-small")
save_fig(fig, "theory_continuum.png")

# Drift waves: Hasegawa-Wakatani growth rates
ky_hw = np.linspace(0.05, 3, 200)
fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
for alpha in (0.1, 1.0, 5.0):
    ax.plot(ky_hw, waves.hasegawa_wakatani(ky_hw, adiabaticity=alpha, gradient=1.0)["drift wave"].imag,
            label=rf"$\alpha$ = {alpha:g}")
ax.set(xlabel=r"$k_y\rho_s$", ylabel=r"$\gamma/\Omega_i$", title=r"Hasegawa–Wakatani drift waves, $\kappa$ = 1")
ax.legend(fontsize="small")
save_fig(fig, "theory_drift_waves.png")

# Orbits in a large-aspect-ratio tokamak: trapped fraction and bounce frequency
fig, (ax_ft, ax_b) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
eps_t = np.linspace(0.0, 0.5, 101)
for approximation, style in (("exact", "-"), ("lin-liu", "--"), ("sqrt", ":")):
    ax_ft.plot(eps_t, orbits.trapped_fraction(eps_t, approximation=approximation), style, label=approximation)
ax_ft.set(xlabel=r"$\varepsilon = r/R_0$", ylabel="trapped fraction", title="Trapped particles")
ax_ft.legend(fontsize="small")
kappa2 = np.linspace(0.0, 0.999, 300)
deep = orbits.bounce_frequency(1.0, 0.0, 0.1, 2.0, 1.0)
ax_b.plot(kappa2, orbits.bounce_frequency(1.0, kappa2, 0.1, 2.0, 1.0) / deep)
ax_b.set(xlabel=r"$\kappa^2$ (0: deeply trapped, 1: trapped-passing boundary)", ylabel=r"$\omega_b/\omega_b(\kappa^2=0)$",
         title="Bounce frequency")
save_fig(fig, "theory_orbits.png")

# Exact solutions: the Sod shock tube and a dam break
x_ex = np.linspace(0, 1, 800)
sod = exact.sod_shock_tube(x_ex, 0.2)
fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
ax_s.plot(x_ex, sod.density, label="density")
ax_s.plot(x_ex, sod.velocity, label="velocity")
ax_s.plot(x_ex, sod.pressure, label="pressure")
ax_s.set(xlabel="x", title="Sod shock tube at t = 0.2 (riemann_euler)")
ax_s.legend(fontsize="small")
x_db = np.linspace(-2, 3, 800)
for t_value in (0.0, 0.3, 0.6, 0.9):
    ax_d.plot(x_db, exact.dam_break(x_db, t_value).depth, label=f"t = {t_value:g}")
ax_d.set(xlabel="x", ylabel="depth", title="Dam break on a dry bed (Ritter)")
ax_d.legend(fontsize="small")
save_fig(fig, "theory_exact.png")

# Numerics: phase errors of time integrators, and the dispersion of spline finite elements
fig, (ax_p, ax_sp) = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
wdt = np.logspace(-2, 0, 100)
for method in ("explicit_euler", "implicit_midpoint", "stormer_verlet", "rk2", "rk4"):
    ax_p.loglog(wdt, np.abs(numerics.phase_error(wdt, method)), label=method)
ax_p.set(xlabel=r"$\omega\,\Delta t$", ylabel="|relative frequency error|", title="Time integrators")
ax_p.legend(fontsize="small")
kdx = np.linspace(0.01, np.pi, 200)
for degree in (1, 2, 3, 4):
    ax_sp.plot(kdx, numerics.spline_galerkin_dispersion(kdx, 1.0, degree).real / kdx, label=f"degree {degree}")
ax_sp.set(xlabel=r"$k\Delta x$", ylabel=r"$\omega_{num}/(ck)$", title="Numerical dispersion of the wave equation")
ax_sp.legend(fontsize="small")
save_fig(fig, "theory_numerics.png")


# =============================================================================
# Whole-run: equilibrium profiles (optional PyVista)
#
# plot_equilibrium_profile/show_equilibrium take a struphy-shaped equilibrium
# (``.p0(eta1, eta2, eta3)``, optionally ``.n0(...)``) and domain mapping
# (``domain(eta1, eta2, eta3, squeeze_out=False) -> (X, Y, Z)``) -- normally
# ``out.equil``/``out.domain`` from a real run. These small duck-typed
# stand-ins keep this figure synthetic without a struphy install.
# =============================================================================
try:
    import pyvista as pv

    from struphy_plots.plotting import (plot_equilibrium_profile,
                                        show_equilibrium)

    pv.OFF_SCREEN = True

    class RadialEquil:
        """A pressure/density profile decreasing from the core outward."""

        def p0(self, eta1, eta2, eta3):
            r = np.meshgrid(eta1, eta2, eta3, indexing="ij")[0]
            return (1 - r**2) ** 1.5 + 0.05

        def n0(self, eta1, eta2, eta3):
            r = np.meshgrid(eta1, eta2, eta3, indexing="ij")[0]
            return (1 - r**2) + 0.1

    def radial_domain(eta1, eta2, eta3, squeeze_out=True):
        """A trivial straight radial line: eta1 *is* the physical radius."""
        r = np.meshgrid(eta1, eta2, eta3, indexing="ij")[0]
        return r, np.zeros_like(r), np.zeros_like(r)

    save(plot_equilibrium_profile(RadialEquil(), radial_domain), "equilibrium.png")

    class ShellEquil:
        """A poloidally-varying pressure on a toroidal boundary shell."""

        def p0(self, eta1, eta2, eta3):
            theta = 2 * np.pi * np.meshgrid(eta1, eta2, eta3, indexing="ij")[1]
            return 1.0 + 0.6 * np.cos(2 * theta)

    def shell_domain(eta1, eta2, eta3, squeeze_out=True):
        """A toroidal shell, thickened slightly along eta1 to avoid a degenerate grid."""
        E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
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
    print(
        f"skipped equilibrium figures (PyVista unavailable or headless rendering failed): {exc}"
    )


# =============================================================================
# 3-D views (optional PyVista): struphy_plots.pyvista_plots
#
# A synthetic torus (the same X/Y/Z-coordinate layout every Struphy field product
# has) with a helical m=3, n=2 mode and a tokamak-like magnetic field; then a 2-D
# run, i.e. a cylinder cross-section whose eta3 direction has a single point.
# =============================================================================
try:
    import pyvista as pv

    from struphy_plots import pyvista_plots as p3

    pv.OFF_SCREEN = True

    def shot(plotter, filename, *, iso=True, zoom=1.0, interactive=True):
        """Save a PNG and, for 3-D scenes, a standalone interactive HTML for <PyVistaScene>."""
        if iso:
            plotter.camera_position = "iso"
            plotter.reset_camera()
        plotter.camera.zoom(zoom)
        plotter.screenshot(str(OUT / filename), window_size=[1000, 700])
        print(f"wrote {OUT / filename}")
        if interactive:
            html = PYVISTA_OUT / filename.replace(".png", ".html")
            # vtk.js receives pre-mapped RGB colors, so an exported color bar would read 0-255;
            # the static image next to each scene keeps the correct one.
            for title in list(plotter.scalar_bars.keys()):
                plotter.remove_scalar_bar(title)
            try:
                plotter.trame.export_html(str(html))
                print(f"wrote {html}")
            except Exception as exc:  # optional: needs trame-pyvista (see .github/workflows/docs.yml)
                print(f"skipped {html.name} (interactive export unavailable): {exc}")
        plotter.close()

    def torus_mapping(eta1, eta2, eta3, squeeze_out=False):
        E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
        r, theta, phi_ = 0.1 + 0.9 * E1, 2 * np.pi * E2, 2 * np.pi * E3
        return (
            (3.0 + r * np.cos(theta)) * np.cos(phi_),
            (3.0 + r * np.cos(theta)) * np.sin(phi_),
            r * np.sin(theta),
        )

    t1, t2, t3 = np.linspace(0, 1, 32), np.linspace(0, 1, 64), np.linspace(0, 1, 96)
    TX, TY, TZ = torus_mapping(t1, t2, t3)
    torus = {
        "eta1": t1, "eta2": t2, "eta3": t3,
        **{n: (("eta1", "eta2", "eta3"), c) for n, c in zip("XYZ", (TX, TY, TZ))},
    }
    TR = 0.1 + 0.9 * t1[:, None, None]
    TTH, TPH = 2 * np.pi * t2[None, :, None], 2 * np.pi * t3[None, None, :]
    t_mode = np.linspace(0.0, 1.0, 16, endpoint=False)
    mode = xr.DataArray(
        np.stack(
            [np.sin(np.pi * TR) ** 2 * np.cos(3 * TTH - 2 * TPH - 2 * np.pi * ti) + 0 * TX for ti in t_mode]
        ),
        dims=("t", "eta1", "eta2", "eta3"),
        coords={"t": t_mode, **torus},
        name="phi",
        attrs={"label": "phi"},
    )
    shot(mode.struphy.plot.isosurface(values=[-0.5, 0.5], cmap="RdBu_r", t=0), "3d_isosurface.png", zoom=1.3)
    shot(
        mode.struphy.plot.slices_3d(cuts={"eta3": [0.0, 0.25, 0.5, 0.75]}, cmap="RdBu_r", t=0),
        "3d_slices.png",
        zoom=1.3,
    )
    shot(mode.struphy.plot.slices_3d(cuts={"eta1": 0.5}, cmap="RdBu_r", t=0), "3d_flux_surface.png", zoom=1.3)

    TRR = np.hypot(TX, TY)
    e_phi = np.stack([-TY / TRR, TX / TRR, 0 * TRR])
    e_theta = np.stack([-np.sin(TTH) * TX / TRR, -np.sin(TTH) * TY / TRR, np.cos(TTH) + 0 * TX])
    b_field = xr.DataArray(
        3.0 / TRR * e_phi + 3.0 * TR / ((1.2 + TR**2) * TRR) * e_theta,
        dims=("component", "eta1", "eta2", "eta3"),
        coords={"component": [0, 1, 2], **torus},
        name="b_field",
        attrs={"label": "B"},
    )
    shot(
        b_field.struphy.plot.streamlines(n_points=60, source_center=(3.5, 0, 0), source_radius=0.35),
        "3d_streamlines.png",
        zoom=1.3,
    )
    shot(b_field.isel(eta1=[24]).struphy.plot.glyphs(stride=3, scale=0.5), "3d_glyphs.png", zoom=1.3)
    def solid_torus(eta1, eta2, eta3, squeeze_out=False):
        """A torus with a polar axis at eta1 = 0, as in most tokamak runs."""
        E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
        theta, phi_ = 2 * np.pi * E2, 2 * np.pi * E3
        return (3.0 + E1 * np.cos(theta)) * np.cos(phi_), (3.0 + E1 * np.cos(theta)) * np.sin(phi_), E1 * np.sin(theta)

    shot(p3.pyvista_domain(solid_torus, n1=6, n2=24, n3=36), "3d_domain.png", zoom=1.3)

    # guiding-center-like orbits: passing markers circle the torus, trapped ones bounce
    rng3 = np.random.default_rng(3)
    n_orb, t_orb = 40, np.linspace(0, 40, 240)[:, None]
    r0, th0, ph0 = rng3.uniform(0.2, 0.8, n_orb), rng3.uniform(0, 2 * np.pi, n_orb), rng3.uniform(0, 2 * np.pi, n_orb)
    trapped3 = rng3.random(n_orb) < 0.4
    bounce = 0.25 * t_orb + th0
    v_par3 = np.where(trapped3, np.cos(bounce), 1.0 + 0 * t_orb)
    path = np.where(trapped3, 1.2 * np.sin(bounce), 0.35 * t_orb)
    th_o, ph_o = th0 + 0.5 * path, ph0 + path
    R_o = 3.0 + r0 * np.cos(th_o)
    xo3, yo3, zo3 = R_o * np.cos(ph_o), R_o * np.sin(ph_o), r0 * np.sin(th_o)
    for arr in (xo3, yo3, zo3, v_par3):
        arr[150:, :2] = 0.0  # two markers leave the domain
    orbits3 = xr.Dataset(
        {n: (("t", "marker"), a) for n, a in zip(("x", "y", "z", "v_par"), (xo3, yo3, zo3, v_par3))},
        coords={"t": t_orb[:, 0], "marker": np.arange(n_orb)},
    )
    shot(
        orbits3.struphy.plot.orbits_3d(color_by="classification", domain=mode.isel(t=0)),
        "3d_orbits.png",
        zoom=1.3,
    )

    # A 2-D run: a cylinder cross-section, eta3 has one point, the plane is z = 0
    c1, c2, c3 = np.linspace(0, 1, 48), np.linspace(0, 1, 96), np.array([0.0])
    C1, C2, _ = np.meshgrid(c1, c2, c3, indexing="ij")
    CR, CTH = 0.05 + 0.95 * C1, 2 * np.pi * C2
    CX, CY = CR * np.cos(CTH), CR * np.sin(CTH)
    cyl = {
        "eta1": c1, "eta2": c2, "eta3": c3,
        **{n: (("eta1", "eta2", "eta3"), c) for n, c in zip("XYZ", (CX, CY, 0 * CX))},
    }
    t_2d = np.linspace(0.0, 1.0, 16, endpoint=False)
    phi_2d = xr.DataArray(
        np.stack([np.sin(np.pi * CR) * np.cos(3 * CTH - 2 * np.pi * ti) for ti in t_2d]),
        dims=("t", "eta1", "eta2", "eta3"),
        coords={"t": t_2d, **cyl},
        name="phi",
        attrs={"label": "phi"},
    )
    shot(phi_2d.struphy.plot.isosurface(values=7, cmap="RdBu_r", t=0), "2d_isosurface.png", iso=False, interactive=False)
    flow_2d = xr.DataArray(
        np.stack([-CY * np.exp(-2 * CR**2), CX * np.exp(-2 * CR**2), 0 * CX])
        + 0.15 * np.stack([np.cos(3 * CTH), np.sin(3 * CTH), 0 * CX]) * np.sin(np.pi * CR),
        dims=("component", "eta1", "eta2", "eta3"),
        coords={"component": [0, 1, 2], **cyl},
        name="u",
        attrs={"label": "u"},
    )
    shot(flow_2d.struphy.plot.streamlines(n_points=80), "2d_streamlines.png", iso=False, interactive=False)

    # GIFs go to public/ so Astro keeps them animated (see PUBLIC_OUT above)
    for movie_data, kind, options, filename in (
        (mode, "slices", {"cuts": {"eta3": [0.0, 0.25, 0.5, 0.75]}, "cmap": "RdBu_r"}, "3d_slices.gif"),
        (phi_2d, "isosurface", {"values": 7, "cmap": "RdBu_r"}, "2d_isosurface.gif"),
    ):
        path = movie_data.struphy.plot.movie(
            PUBLIC_OUT / filename, kind=kind, framerate=8, window_size=(800, 560), **options
        )
        print(f"wrote {path}")
    # The filtered TAE from the spectral-analysis section, in its torus sector
    sector_R = 10.0 + r_tae * np.cos(2 * np.pi * S2)
    sector = {
        "X": (("eta1", "eta2", "eta3"), sector_R * np.cos(2 * np.pi * S3 / 6)),
        "Y": (("eta1", "eta2", "eta3"), sector_R * np.sin(2 * np.pi * S3 / 6)),
        "Z": (("eta1", "eta2", "eta3"), r_tae * np.sin(2 * np.pi * S2)),
    }
    tae_3d = band_tae.filtered.isel(t=-1).assign_coords(sector)
    shot(tae_3d.struphy.plot.slices_3d(cuts={"eta3": [0.0, 0.5, 1.0], "eta1": 0.44}, cmap="RdBu_r"), "spectral_tae_3d.png", zoom=1.2)
except Exception as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped 3-D view figures (PyVista unavailable or headless rendering failed): {exc}")


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
    with _sp.session(
        verbose=False, file_path=str(Path(_profiling_tmp) / "profiling_data.h5")
    ):
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


# =============================================================================
# backend="plotly": the same plots as interactive Plotly figures (needs `pip install plotly`),
# as figure JSON for the docs' <PlotlyChart> component. The folds "The same plot with Plotly"
# in the guides show these, with the same arguments as the Matplotlib figures above.
# =============================================================================
try:
    import plotly  # noqa: F401

    def save_plotly(result, filename):
        path = result.save(PLOTLY_OUT / f"{filename}.json")
        print(f"wrote {path}")

    save_plotly(field.struphy.plot.slice(x="eta1", y="eta2", t=-1, backend="plotly"), "plotly_view_slice")
    save_plotly(vector.struphy.plot.vector(x="eta1", y="eta2", components=(0, 1), stride=6, t=-1, backend="plotly"),
                "plotly_vector")
    save_plotly(volume_data.struphy.plot.volume_slices(backend="plotly"), "plotly_volume_slices")
    save_plotly(energy.struphy.plot.timeseries(fit=(0.0, 2.0), title="Field energy growth", backend="plotly"),
                "plotly_timeseries")
    save_plotly(run_a.struphy.plot.compare(run_b, mode="ratio", backend="plotly"), "plotly_compare")
    save_plotly(distribution.struphy.plot.slice(x="eta1", y="v1", t=-1, backend="plotly"), "plotly_phase_space")
    save_plotly(marker_orbits.struphy.plot.trajectories(show_paths=True, backend="plotly"), "plotly_trajectories")
    save_plotly(cloud.struphy.plot.scatter(x="x", y="y", color="density", backend="plotly"), "plotly_scatter")
    save_plotly(well.struphy.plot.overlay_orbits(confined_orbits, x="eta1", y="eta2", backend="plotly"),
                "plotly_overlay_orbits")
    save_plotly(dispersive_field.struphy.plot.dispersion(branches={"Bohm-Gross": bohm_gross}, kmax=7, omega_max=12,
                                                         backend="plotly"), "plotly_dispersion")
    save_plotly(light_spectrum.struphy.plot.dispersion(kmin=0, branches={"light, ω = k": lambda k: k}, fits=light_fits,
                                                       dynamic_range=12, omega_max=25, backend="plotly"), "plotly_dispersion_fits")
    # the Plotly guide
    save_plotly(ring.struphy.plot.slice(coords="physical", plane="XY", t=-1, eta3=0, levels=[0.2], backend="plotly"),
                "plotly_ring_slice")
    save_plotly(ring.struphy.plot.animation(coords="physical", plane="XY", eta3=0, levels=[0.2], step=4,
                                            backend="plotly"), "plotly_ring_animation")
    save_plotly(phi_tae.struphy.plot.power_spectrum(peaks=2, band=band_tae, frequencies={"TAE gap-centre estimate":
                                                    omega_tae}, omega_max=0.5, backend="plotly"),
                "plotly_spectral_power")
except ImportError as exc:  # pragma: no cover - optional, environment-dependent
    print(f"skipped the backend=\"plotly\" figures (plotly unavailable): {exc}")

plt.close("all")
print("done")
