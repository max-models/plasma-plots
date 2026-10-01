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
    values = np.stack(
        [
            np.exp(0.1 * ti) * E1 * np.cos(2 * np.pi * (3 * E2 - E3) - 0.8 * ti)
            for ti in t
        ]
    )
    return xr.DataArray(
        values,
        dims=("t", "eta1", "eta2", "eta3"),
        coords=coords,
        name="phi",
        attrs={"label": r"$\phi$", "run": "synthetic run"},
    )


def wave():
    t, x = np.linspace(0, 60, 200), np.linspace(0, 1, 16, endpoint=False)
    values = np.cos(2 * np.pi * 3 * x[None] - 0.9 * t[:, None]) + 0.3 * np.cos(
        2 * np.pi * x[None] - 0.3 * t[:, None]
    )
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
        },
        coords={"t": t, "marker": np.arange(nm)},
        attrs={"label": "ions"},
    )


def cases():
    phi, u, e, o = torus_field(), wave(), energy(), orbits()
    probe = u.isel(eta1=3)
    vec = (
        xr.concat([phi, 0.5 * phi], dim="component")
        .transpose("t", "component", ...)
        .rename("E")
    )
    k = np.linspace(0.5, 3, 6)
    measured = xr.DataArray(
        np.sqrt(1 + 3 * k**2) * 1.01, dims="k", coords={"k": k}, name="omega"
    )
    band = phi.plasma.analysis.filter_time(dims=("eta1", "eta2", "eta3"))
    other = e * 1.5
    return {
        "timeseries": lambda **b: e.plasma.plot.timeseries(
            other, fit=(2.0, 8.0), reference=lambda t: 1e-4 * np.exp(0.4 * t), **b
        ),
        "lineout": lambda **b: phi.plasma.plot.lineout(
            x="eta1", t=-1, eta2=0, eta3=0, reference=lambda x: x, **b
        ),
        "line_animation": lambda **b: phi.plasma.plot.line_animation(
            x="eta1", eta2=0, eta3=0, **b
        ),
        "against_theory": lambda **b: measured.plasma.plot.against_theory(
            lambda k: np.sqrt(1 + 3 * k**2), **b
        ),
        "vector": lambda **b: vec.plasma.plot.vector(
            x="eta1", y="eta2", t=-1, eta3=0, **b
        ),
        "volume_slices": lambda **b: phi.plasma.plot.volume_slices(t=-1, **b),
        "compare": lambda **b: phi.isel(eta2=0, eta3=0, t=-1).plasma.plot.compare(
            phi.isel(eta2=0, eta3=0, t=0), **b
        ),
        "overlay_orbits": lambda **b: phi.plasma.plot.overlay_orbits(
            o, x="eta1", y="eta2", t=-1, eta3=0, **b
        ),
        "dispersion": lambda **b: u.plasma.plot.dispersion(
            dim="eta1",
            branches={"w": lambda k: 0.3 * np.abs(k)},
            frequencies={"cut": 0.5},
            **b,
        ),
        "power_spectrum": lambda **b: probe.plasma.plot.power_spectrum(
            peaks=2, band=(0.8, 1.0), frequencies={"x": 0.3}, **b
        ),
        "filtered": lambda **b: phi.plasma.plot.filtered(
            band, eta1=0.5, eta2=0.0, eta3=0.0, **b
        ),
        "spectrogram": lambda **b: probe.plasma.plot.spectrogram(
            length=100, step=20, **b
        ),
        "mode_amplitudes": lambda **b: phi.plasma.plot.mode_amplitudes(
            top=2, fit=True, **b
        ),
        "mode_map": lambda **b: phi.plasma.plot.mode_map(t=-1, **b),
        "radial_power": lambda **b: phi.plasma.plot.radial_power(**b),
        "mode_profiles": lambda **b: phi.plasma.plot.mode_profiles(0.8, top=2, **b),
        "profiles": lambda **b: phi.plasma.plot.profiles(x="eta1", eta2=0, eta3=0, **b),
        "cross_spectrum": lambda **b: u.plasma.plot.cross_spectrum(
            u.roll(eta1=2), dims="eta1", **b
        ),
        "pencil_fit": lambda **b: probe.sel(t=slice(0, 20)).plasma.plot.pencil_fit(
            n_modes=2, **b
        ),
        "slice": lambda **b: phi.plasma.plot.slice(
            x="eta1", y="eta2", t=-1, eta3=0, levels=4, **b
        ),
        "slice_physical": lambda **b: phi.plasma.plot.slice(
            coords="physical",
            plane="RZ",
            t=-1,
            eta3=0,
            overlays={"boundary": True, "points": {"o": (3.0, 0.0)}},
            **b,
        ),
        "panels": lambda **b: phi.plasma.plot.panels(
            x="eta1", y="eta2", nrows=1, ncols=2, eta3=0, **b
        ),
        "viewer": lambda **b: phi.plasma.plot.viewer(x="eta1", y="eta2", eta3=0, **b),
        "animation": lambda **b: phi.plasma.plot.animation(
            coords="physical", plane="RZ", eta3=0, **b
        ),
        "trajectories": lambda **b: o.plasma.plot.trajectories(**b),
        "view_slice": lambda **b: phi.plasma.plot.view(
            x="eta1", y="eta2", eta3=0
        ).slice(t=-1, **b),
        "view_panels": lambda **b: phi.plasma.plot.view(
            x="eta1", y="eta2", eta3=0
        ).panels(nrows=1, ncols=2, **b),
        "view_viewer": lambda **b: phi.plasma.plot.view(
            x="eta1", y="eta2", eta3=0
        ).viewer(**b),
        "view_animation": lambda **b: phi.plasma.plot.view(
            x="eta1", y="eta2", eta3=0
        ).animation(alongside=[phi**2], **b),
        "ds_power_spectrum": lambda **b: probe.plasma.analysis.time_fft().plasma.plot.power_spectrum(
            peaks=1, **b
        ),
        "ds_cross_spectrum": lambda **b: u.plasma.analysis.cross_spectrum(
            u.roll(eta1=2), dims="eta1"
        ).plasma.plot.cross_spectrum(**b),
        "ds_trajectories": lambda **b: o.plasma.plot.trajectories(**b),
        "ds_scatter": lambda **b: o.plasma.plot.scatter(
            x="eta1", y="eta2", color="v_par", t=-1, background=phi.isel(eta3=0), **b
        ),
        "ds_orbit_classification": lambda **b: o.plasma.plot.orbit_classification(**b),
        "ds_animation": lambda **b: o.plasma.plot.animation(
            x="eta1", y="eta2", color="v_par", step=10, **b
        ),
        "ds_paths": lambda **b: o.plasma.plot.paths(
            x="eta1", y="eta2", markers=2, background=phi.isel(eta3=0), **b
        ),
        "ds_poloidal": lambda **b: o.plasma.plot.poloidal(
            color_by="t", boundary=phi.isel(t=0), **b
        ),
        "ds_orbit_grid": lambda **b: o.plasma.plot.orbit_grid(markers=2, ncols=2, **b),
        "ds_quantities": lambda **b: o.plasma.plot.quantities(markers=2, **b),
    }


CASES = cases()
