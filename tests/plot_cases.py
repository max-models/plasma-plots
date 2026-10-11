"""Every accessor plot on small synthetic data, as functions of the backend option.

Shared by the backend tests: ``CASES[name](backend="plotly")`` draws plot ``name``.
"""

import numpy as np
import xarray as xr

import plasma_plots  # noqa: F401  (registers the .plasma accessors)


def torus_field(nt=5, n1=6, n2=12, n3=4):
    """(t, eta1, eta2, eta3) on a hollow torus, with its X, Y, Z."""
    t = np.linspace(0, 4, nt)
    eta1, eta2, eta3 = (
        np.linspace(0.1, 1, n1),
        np.linspace(0, 1, n2, endpoint=False),
        np.linspace(0, 1, n3, endpoint=False),
    )
    E1, E2, E3 = np.meshgrid(eta1, eta2, eta3, indexing="ij")
    R = 3 + E1 * np.cos(2 * np.pi * E2)
    coords = {
        "t": t,
        "eta1": eta1,
        "eta2": eta2,
        "eta3": eta3,
        "X": (("eta1", "eta2", "eta3"), R * np.cos(2 * np.pi * E3)),
        "Y": (("eta1", "eta2", "eta3"), R * np.sin(2 * np.pi * E3)),
        "Z": (("eta1", "eta2", "eta3"), E1 * np.sin(2 * np.pi * E2)),
    }
    values = np.stack([np.exp(0.1 * ti) * E1 * np.cos(2 * np.pi * (3 * E2 - E3) - 0.8 * ti) for ti in t])
    return xr.DataArray(
        values,
        dims=("t", "eta1", "eta2", "eta3"),
        coords=coords,
        name="phi",
        attrs={"label": r"$\phi$", "run": "synthetic run"},
    )


def wave():
    t, x = np.linspace(0, 60, 200), np.linspace(0, 1, 16, endpoint=False)
    values = np.cos(2 * np.pi * 3 * x[None] - 0.9 * t[:, None]) + 0.3 * np.cos(2 * np.pi * x[None] - 0.3 * t[:, None])
    return xr.DataArray(
        values,
        dims=("t", "eta1"),
        coords={"t": t, "eta1": x},
        name="u",
        attrs={"label": "u"},
    )


def energy():
    t = np.linspace(0, 10, 101)
    return xr.DataArray(
        1e-4 * np.exp(0.4 * t) * (1 + 0.1 * np.cos(3 * t)),
        dims="t",
        coords={"t": t},
        name="en_E",
        attrs={"label": "$E_x$ energy"},
    )


def orbits(nt=30, nm=5):
    t = np.linspace(0, 4, nt)
    angle = t[:, None] * (1 + np.arange(nm))[None]
    return xr.Dataset(
        {
            "x": (("t", "marker"), (3 + 0.3 * np.cos(angle)) * np.cos(0.2 * angle)),
            "y": (("t", "marker"), (3 + 0.3 * np.cos(angle)) * np.sin(0.2 * angle)),
            "z": (("t", "marker"), 0.3 * np.sin(angle)),
            "eta1": (("t", "marker"), 0.5 + 0.3 * np.cos(angle)),
            "eta2": (("t", "marker"), (0.5 + 0.3 * np.sin(angle)) % 1),
            "v_par": (("t", "marker"), np.cos(angle) + np.linspace(-1, 1.5, nm)),
            "mu": (("t", "marker"), 0.1 + 0 * angle + np.arange(nm) * 0.05),
            "weight": (("t", "marker"), 1e-3 * np.sin(angle) * (1 + t[:, None])),
        },
        coords={"t": t, "marker": np.arange(nm)},
        attrs={"label": "ions"},
    )


def tokamak_field(n1=10, n2=24, n3=12, radial=0.0):
    """A tokamak-like B over (component, eta1, eta2, eta3) on a torus: field lines with iota = 1/q;
    ``radial`` adds an outward component, so that the lines leave through the edge."""
    e1, e2, e3 = (
        np.linspace(0, 1, n1),
        (np.arange(n2) + 0.5) / n2,
        (np.arange(n3) + 0.5) / n3,
    )
    E1, E2, E3 = np.meshgrid(e1, e2, e3, indexing="ij")
    r, th, ph = 0.1 + 0.9 * E1, 2 * np.pi * E2, 2 * np.pi * E3
    R = 3 + r * np.cos(th)
    e_r = np.stack([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), np.sin(th)])
    e_th = np.stack([-np.sin(th) * np.cos(ph), -np.sin(th) * np.sin(ph), np.cos(th)])
    e_ph = np.stack([-np.sin(ph), np.cos(ph), 0 * ph])
    B = 3 / R * e_ph + r / ((1 + 2 * r**2) * 3) * e_th + radial * e_r
    return xr.DataArray(
        B,
        dims=("component", "eta1", "eta2", "eta3"),
        coords={
            "component": [0, 1, 2],
            "eta1": e1,
            "eta2": e2,
            "eta3": e3,
            "X": (("eta1", "eta2", "eta3"), R * np.cos(ph)),
            "Y": (("eta1", "eta2", "eta3"), R * np.sin(ph)),
            "Z": (("eta1", "eta2", "eta3"), r * np.sin(th)),
        },
        name="B",
        attrs={"label": "$B$"},
    )


def flux_box(n=24):
    """A tearing-like flux function over a periodic box, growing in time."""
    e = (np.arange(n) + 0.5) / n
    E1, E2 = np.meshgrid(e, e, indexing="ij")
    X, Y = 2 * np.pi * E1, 2 * np.pi * E2
    t = np.linspace(0, 2, 3)
    values = np.stack([-np.cos(Y - 0.3) + 0.2 * np.exp(ti) * np.cos(X - 0.7) for ti in t])
    return xr.DataArray(
        values[..., None],
        dims=("t", "eta1", "eta2", "eta3"),
        coords={
            "t": t,
            "eta1": ("eta1", e, {"period": 1.0}),
            "eta2": ("eta2", e, {"period": 1.0}),
            "eta3": [0.0],
            "X": (("eta1", "eta2", "eta3"), X[..., None]),
            "Y": (("eta1", "eta2", "eta3"), Y[..., None]),
            "Z": (("eta1", "eta2", "eta3"), 0 * X[..., None]),
        },
        name="A",
        attrs={"label": "$A$"},
    )


def boozer_field():
    """|B| on a Boozer grid with nfp 3: axisymmetric plus one symmetry-breaking harmonic."""
    rho = np.linspace(0.2, 1, 5)
    theta = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / 3, 12, endpoint=False)
    R, TH, ZE = np.meshgrid(rho, theta, zeta, indexing="ij")
    return xr.DataArray(
        1 + 0.1 * R * np.cos(TH) + 0.02 * R**2 * np.cos(2 * TH - 3 * ZE),
        dims=("rho", "theta_B", "zeta_B"),
        coords={
            "rho": rho,
            "theta_B": ("theta_B", theta, {"period": 2 * np.pi}),
            "zeta_B": ("zeta_B", zeta, {"period": 2 * np.pi / 3}),
        },
        name="mod_B",
        attrs={"label": "$|B|$", "nfp": 3},
    )


def cases():
    phi, u, e, o = torus_field(), wave(), energy(), orbits()
    B = tokamak_field()
    lines = B.plasma.analysis.field_lines(seeds=4, turns=6)
    open_lines = tokamak_field(radial=0.3).plasma.analysis.field_lines(
        seeds={"eta1": 0.5, "eta2": np.linspace(0.1, 0.9, 3), "eta3": [0.1, 0.5]},
        direction="both",
        length=40.0,
    )
    flux = flux_box()
    probe = u.isel(eta1=3)
    vec = xr.concat([phi, 0.5 * phi], dim="component").transpose("t", "component", ...).rename("E")
    k = np.linspace(0.5, 3, 6)
    measured = xr.DataArray(np.sqrt(1 + 3 * k**2) * 1.01, dims="k", coords={"k": k}, name="omega")
    band = phi.plasma.analysis.filter_time(dims=("eta1", "eta2", "eta3"))
    other = e * 1.5
    return {
        "timeseries": lambda **b: e.plasma.plot.timeseries(
            other, fit=(2.0, 8.0), reference=lambda t: 1e-4 * np.exp(0.4 * t), **b
        ),
        "lineout": lambda **b: phi.plasma.plot.lineout(x="eta1", t=-1, eta2=0, eta3=0, reference=lambda x: x, **b),
        "line_animation": lambda **b: phi.plasma.plot.line_animation(x="eta1", eta2=0, eta3=0, **b),
        "against_theory": lambda **b: measured.plasma.plot.against_theory(lambda k: np.sqrt(1 + 3 * k**2), **b),
        "vector": lambda **b: vec.plasma.plot.vector(x="eta1", y="eta2", t=-1, eta3=0, **b),
        "volume_slices": lambda **b: phi.plasma.plot.volume_slices(t=-1, **b),
        "compare": lambda **b: phi.isel(eta2=0, eta3=0, t=-1).plasma.plot.compare(phi.isel(eta2=0, eta3=0, t=0), **b),
        "overlay_orbits": lambda **b: phi.plasma.plot.overlay_orbits(o, x="eta1", y="eta2", t=-1, eta3=0, **b),
        "dispersion": lambda **b: u.plasma.plot.dispersion(
            dim="eta1",
            branches={"w": lambda k: 0.3 * np.abs(k)},
            frequencies={"cut": 0.5},
            **b,
        ),
        "power_spectrum": lambda **b: probe.plasma.plot.power_spectrum(
            peaks=2, band=(0.8, 1.0), frequencies={"x": 0.3}, **b
        ),
        "filtered": lambda **b: phi.plasma.plot.filtered(band, eta1=0.5, eta2=0.0, eta3=0.0, **b),
        "spectrogram": lambda **b: probe.plasma.plot.spectrogram(length=100, step=20, **b),
        "mode_amplitudes": lambda **b: phi.plasma.plot.mode_amplitudes(top=2, fit=True, **b),
        "mode_map": lambda **b: phi.plasma.plot.mode_map(t=-1, **b),
        "radial_power": lambda **b: phi.plasma.plot.radial_power(**b),
        "mode_profiles": lambda **b: phi.plasma.plot.mode_profiles(0.8, top=2, **b),
        "profiles": lambda **b: phi.plasma.plot.profiles(x="eta1", eta2=0, eta3=0, **b),
        "cross_spectrum": lambda **b: u.plasma.plot.cross_spectrum(u.roll(eta1=2), dims="eta1", **b),
        "pencil_fit": lambda **b: probe.sel(t=slice(0, 20)).plasma.plot.pencil_fit(n_modes=2, **b),
        "slice": lambda **b: phi.plasma.plot.slice(x="eta1", y="eta2", t=-1, eta3=0, levels=4, **b),
        "slice_physical": lambda **b: phi.plasma.plot.slice(
            coords="physical",
            plane="RZ",
            t=-1,
            eta3=0,
            overlays={"boundary": True, "points": {"o": (3.0, 0.0)}},
            **b,
        ),
        "panels": lambda **b: phi.plasma.plot.panels(x="eta1", y="eta2", nrows=1, ncols=2, eta3=0, **b),
        "viewer": lambda **b: phi.plasma.plot.viewer(x="eta1", y="eta2", eta3=0, **b),
        "animation": lambda **b: phi.plasma.plot.animation(coords="physical", plane="RZ", eta3=0, **b),
        "trajectories": lambda **b: o.plasma.plot.trajectories(**b),
        "view_slice": lambda **b: phi.plasma.plot.view(x="eta1", y="eta2", eta3=0).slice(t=-1, **b),
        "view_panels": lambda **b: phi.plasma.plot.view(x="eta1", y="eta2", eta3=0).panels(nrows=1, ncols=2, **b),
        "view_viewer": lambda **b: phi.plasma.plot.view(x="eta1", y="eta2", eta3=0).viewer(**b),
        "view_animation": lambda **b: phi.plasma.plot.view(x="eta1", y="eta2", eta3=0).animation(
            alongside=[phi**2], **b
        ),
        "ds_power_spectrum": lambda **b: probe.plasma.analysis.time_fft().plasma.plot.power_spectrum(peaks=1, **b),
        "ds_cross_spectrum": lambda **b: u.plasma.analysis.cross_spectrum(
            u.roll(eta1=2), dims="eta1"
        ).plasma.plot.cross_spectrum(**b),
        "ds_trajectories": lambda **b: o.plasma.plot.trajectories(**b),
        "ds_scatter": lambda **b: o.plasma.plot.scatter(
            x="eta1", y="eta2", color="v_par", t=-1, background=phi.isel(eta3=0), **b
        ),
        "ds_orbit_classification": lambda **b: o.plasma.plot.orbit_classification(**b),
        "ds_animation": lambda **b: o.plasma.plot.animation(x="eta1", y="eta2", color="v_par", step=10, **b),
        "ds_paths": lambda **b: o.plasma.plot.paths(x="eta1", y="eta2", markers=2, background=phi.isel(eta3=0), **b),
        "ds_poloidal": lambda **b: o.plasma.plot.poloidal(color_by="t", boundary=phi.isel(t=0), **b),
        "ds_orbit_grid": lambda **b: o.plasma.plot.orbit_grid(markers=2, ncols=2, **b),
        "ds_quantities": lambda **b: o.plasma.plot.quantities(markers=2, **b),
        "poincare": lambda **b: B.plasma.plot.poincare(seeds=3, turns=4, **b),
        "surface_map": lambda **b: phi.plasma.plot.surface_map(
            eta1=0.5, t=-1, iota=0.7, count=3, lines=lines.isel(line=[0]), **b
        ),
        "along_field_lines": lambda **b: phi.plasma.plot.along_field_lines(lines, k_parallel=True, t=-1, **b),
        "critical_points": lambda **b: flux.plasma.plot.critical_points(t=-1, eta3=0, label_values=True, **b),
        "critical_points_physical": lambda **b: flux.plasma.plot.critical_points(coords="physical", t=-1, eta3=0, **b),
        "boozer_spectrum": lambda **b: boozer_field().plasma.plot.boozer_spectrum(top=3, helicity="QA", **b),
        "ds_poincare": lambda **b: lines.plasma.plot.poincare(
            color_by="classification", islands=True, boundary=phi.isel(t=0), **b
        ),
        "ds_poincare_iota": lambda **b: lines.plasma.plot.poincare(coords="logical", color_by="iota", **b),
        "ds_field_lines": lambda **b: lines.plasma.plot.field_lines(plane="RZ", color_by="absB", **b),
        "ds_footprint": lambda **b: open_lines.plasma.plot.footprint(**b),
        "ds_connection_length": lambda **b: open_lines.plasma.plot.connection_length(**b),
        "ds_weight_histogram": lambda **b: o.plasma.plot.weight_histogram(t=[0, -1], bins=8, **b),
        "ds_marker_density": lambda **b: o.plasma.plot.marker_density(
            x="eta1", bins=6, against=phi.isel(t=0, eta2=0, eta3=0), **b
        ),
        "ds_lost_fraction": lambda **b: o.plasma.plot.lost_fraction(**b),
        "ds_loss_map": lambda **b: o.plasma.plot.loss_map(**b),
    }


CASES = cases()
