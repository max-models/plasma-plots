"""Generate the figures for the "A real simulation" guide (docs/src/assets/figures/real_*).

Every other example figure in the docs uses small synthetic data (see
``generate_docs_figures.py``) so contributors can build the docs without a full struphy
install. This script is the one exception: it's a real struphy simulation, adapted from
struphy's own gallery example (mhd-slab-waves), run right here -- so it needs the full
compiled struphy runtime (see .github/workflows/docs.yml and CONTRIBUTING.md for the
system packages/`struphy compile` step).

Run from the repo root: python scripts/generate_real_example_figures.py
(or: make figures, which runs this and generate_docs_figures.py)
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from scipy.signal import argrelextrema

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import struphy_plots  # noqa: F401  (registers .struphy on DataArray/Dataset)

DOCS = Path(__file__).resolve().parents[1] / "docs"
OUT = DOCS / "src" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# A real magnetized slab: struphy.models.LinearMHD, in a uniform, obliquely
# magnetized plasma (struphy.fields_background.equils.HomogenSlab). Broadband
# noise in the velocity excites all three MHD wave branches along z at once:
# the shear Alfvén wave, and the slow and fast magnetosonic waves.
# Adapted from struphy's own gallery example (mhd-slab-waves).
# =============================================================================
from struphy import DerhamOptions, EnvironmentOptions, Time, domains, equils, grids, perturbations
from struphy.models import LinearMHD
from struphy.simulation.sim import Simulation

# The background: B0 = (0, 1, 1), density 0.7, plasma beta 3 (thermal over magnetic pressure).
B0x, B0y, B0z = 0.0, 1.0, 1.0
n0, beta, gamma = 0.7, 3.0, 5.0 / 3.0
B_squared = B0x**2 + B0y**2 + B0z**2
p0 = beta * B_squared / 2.0

# The exact ideal-MHD wave speeds along z, to compare the measured ones against.
alfven_speed = np.sqrt(B_squared / n0)
sound_speed = np.sqrt(gamma * p0 / n0)
delta = 4 * B0z**2 * sound_speed**2 * alfven_speed**2 / ((sound_speed**2 + alfven_speed**2) ** 2 * B_squared)
exact_speeds = {
    "shear Alfven": alfven_speed * B0z / np.sqrt(B_squared),
    "slow magnetosonic": np.sqrt(0.5 * (sound_speed**2 + alfven_speed**2) * (1.0 - np.sqrt(1.0 - delta))),
    "fast magnetosonic": np.sqrt(0.5 * (sound_speed**2 + alfven_speed**2) * (1.0 + np.sqrt(1.0 - delta))),
}

model = LinearMHD()
model.propagators.shear_alf.options = model.propagators.shear_alf.Options(algo="implicit")
for component in range(3):
    model.mhd.velocity.add_perturbation(perturbations.Noise(amp=0.1, comp=component, seed=123))
equil = equils.HomogenSlab(B0x=B0x, B0y=B0y, B0z=B0z, beta=beta, n0=n0)

sim = Simulation(
    model=model,
    env=EnvironmentOptions(out_folders=tempfile.mkdtemp(), sim_folder="mhd_slab_waves"),
    time_opts=Time(dt=0.15, Tend=180.0),
    domain=domains.Cuboid(r3=60.0),
    grid=grids.TensorProductGrid(num_elements=(1, 1, 64)),
    derham_opts=DerhamOptions(degree=(1, 1, 3)),
    equil=equil,
)
out = sim.run()

def fitted_branch(spectrum, *, n_branches, noise_level, order=10):
    """Fit omega = v * k to each of ``n_branches`` ridges in a (omega, k) power spectrum,
    by finding local maxima along omega at each k and fitting their locus -- adapted from
    struphy's own test suite (models/tests/verification/test_verif_LinearMHD.py)."""
    omega_mask, k_mask = spectrum.omega.values >= 0, spectrum.k.values >= 0
    power = spectrum.values[np.ix_(omega_mask, k_mask)]
    omega, k = spectrum.omega.values[omega_mask], spectrum.k.values[k_mask]

    k_fit, peaks_fit = [], [[] for _ in range(n_branches)]
    for i in range(k.size // 8, k.size // 2):
        column = power[:, i]
        maxima = argrelextrema(column, np.greater, order=order)[0]
        peaks = sorted(m for m in maxima if column[m] > noise_level * column.max())
        if len(peaks) != n_branches:
            continue
        k_fit.append(k[i])
        for branch, m in zip(peaks_fit, peaks):
            branch.append(omega[m])
    return [np.polyfit(k_fit, branch, deg=1)[0] for branch in peaks_fit]


# Both fields are evaluated at a single (e1, e2) point, physical z as the remaining spatial
# coordinate (logical e3 swapped for physical Z, so k comes out in physical units, matching
# the exact speeds above).
velocity = out.evaluate("mhd/velocity", component=0).isel(e1=0, e2=0)
velocity = velocity.assign_coords(e3=("e3", velocity["Z"].values))
pressure = out.evaluate("mhd/pressure").isel(e1=0, e2=0)
pressure = pressure.assign_coords(e3=("e3", pressure["Z"].values))

velocity_spectrum = velocity.struphy.analysis.dispersion(dim="e3")
pressure_spectrum = pressure.struphy.analysis.dispersion(dim="e3")

(measured_alfven,) = fitted_branch(velocity_spectrum, n_branches=1, noise_level=0.5)
measured_slow, measured_fast = fitted_branch(pressure_spectrum, n_branches=2, noise_level=0.4)
measured_speeds = {"shear Alfven": measured_alfven, "slow magnetosonic": measured_slow, "fast magnetosonic": measured_fast}
for branch, exact in exact_speeds.items():
    print(f"{branch}: measured {measured_speeds[branch]:.4f}, exact {exact:.4f}")

kmax, omega_max = 0.5, 1.3 * exact_speeds["fast magnetosonic"] * 0.5

velocity_path = OUT / "real_dispersion_velocity.png"
velocity.struphy.plot.dispersion(
    branches={
        "shear Alfven (exact)": lambda k: exact_speeds["shear Alfven"] * k,
        "shear Alfven (measured)": lambda k: measured_alfven * k,
    },
    kmax=kmax,
    omega_max=omega_max,
).save(velocity_path, close=True)
print(f"wrote {velocity_path}")

pressure_path = OUT / "real_dispersion_pressure.png"
pressure.struphy.plot.dispersion(
    branches={
        "slow (exact)": lambda k: exact_speeds["slow magnetosonic"] * k,
        "slow (measured)": lambda k: measured_slow * k,
        "fast (exact)": lambda k: exact_speeds["fast magnetosonic"] * k,
        "fast (measured)": lambda k: measured_fast * k,
    },
    kmax=kmax,
    omega_max=omega_max,
).save(pressure_path, close=True)
print(f"wrote {pressure_path}")

print("done")
