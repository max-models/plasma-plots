"""Gradient-drift instability

The gradient-drift (Simon–Hoh) instability in a magnetron: growth rate of the instability of a crossed-field argon plasma, from the cold-electron Simon–Hoh relation and the kinetic-corrected dispersion relation.

The instability needs the electric field and the density gradient to point the same way
($E_0 L_n > 0$). The Simon–Hoh growth rate rises with $k$ and sets the threshold $E_0/(B\\,\\Omega_i L_n) > 1/4$.
The full relation follows it at long wavelength; the finite electron Larmor radius cuts the growth off at
short wavelength (stable beyond $k \\approx 2500$ m$^{-1}$ without collisions), and electron–neutral collisions
(here $\\nu$ up to $10^{10}$ s$^{-1}$) damp the peak and flatten it.
"""

import matplotlib.pyplot as plt
import numpy as np

from plasma_plots.theory import magnetron

e0, b, ln, mass = 1e4, 0.05, 0.01, 39.948  # V/m, T, m, argon
k = np.linspace(10, 3000, 400)  # 1/m

fig, (ax_k, ax_e) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
ax_k.plot(
    k,
    magnetron.simon_hoh_growth(k, e0, b, ln, mass) * 1e-6,
    "k--",
    lw=2,
    label="Simon–Hoh",
)
for nu in (0.0, 1e8, 1e9, 1e10):
    gamma = magnetron.gradient_drift_growth(
        k, e0, b, ln, 3.0, mass, collision_frequency=nu
    )
    ax_k.plot(k, gamma * 1e-6, lw=2, label=rf"full, $\nu$ = {nu:.0e} s$^{{-1}}$")
ax_k.set(
    xlabel="$k$ (1/m)",
    ylabel=r"growth rate $\gamma$ (10$^6$ s$^{-1}$)",
    title="Growth vs wavenumber",
)
ax_k.legend(frameon=False)

k0 = 2 * np.pi / 0.05
e_axis = np.linspace(-2e4, 2e4, 400)
ax_e.plot(
    e_axis * 1e-3,
    magnetron.simon_hoh_growth(k0, e_axis, b, ln, mass) * 1e-6,
    "k--",
    lw=2,
    label="Simon–Hoh",
)
ax_e.plot(
    e_axis * 1e-3,
    magnetron.gradient_drift_growth(k0, e_axis, b, ln, 3.0, mass) * 1e-6,
    lw=2,
    label="full, $T_e$ = 3 eV",
)
ax_e.set(
    xlabel=r"$E_0$ (kV/m), $L_n > 0$",
    ylabel=r"growth rate $\gamma$ (10$^6$ s$^{-1}$)",
    title=r"Growth vs electric field, $\lambda$ = 5 cm",
)
ax_e.legend(frameon=False)

gamma0 = magnetron.gradient_drift_growth(k0, e0, b, ln, 3.0, mass)
print(f"E0 = 10 kV/m, lambda = 5 cm: gamma = {gamma0:.3e} 1/s")
plt.show()
