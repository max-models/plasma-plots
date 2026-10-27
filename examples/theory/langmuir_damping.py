"""Landau damping of a Langmuir wave

Frequency and damping rate of a Langmuir wave in a Maxwellian plasma, against the Bohm–Gross fluid result.

The kinetic dispersion relation $\\epsilon(k, \\omega) = 0$ is solved with the plasma dispersion function.
For $k\\lambda_D \\ll 1$ the wave is nearly undamped and follows Bohm–Gross; the damping grows
exponentially with $k\\lambda_D$ and the wave is gone by $k\\lambda_D \\approx 0.6$.
"""

import matplotlib.pyplot as plt
import numpy as np

from plasma_plots.theory import kinetic

k = np.linspace(0.1, 0.6, 60)
omega = kinetic.langmuir(k)

fig, (ax_f, ax_g) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
ax_f.plot(k, omega.real, lw=2, label="kinetic")
ax_f.plot(k, kinetic.bohm_gross(k).real, "--", lw=1.5, label="Bohm–Gross")
ax_f.set(
    xlabel=r"$k\lambda_D$", ylabel=r"$\mathrm{Re}\,\omega/\omega_p$", title="Frequency"
)
ax_f.legend(frameon=False)

ax_g.semilogy(k, -omega.imag, lw=2)
ax_g.set(
    xlabel=r"$k\lambda_D$",
    ylabel=r"$-\mathrm{Im}\,\omega/\omega_p$",
    title="Landau damping rate",
)

print(f"k lambda_D = 0.5: omega = {kinetic.langmuir(0.5):.4f}")
plt.show()
