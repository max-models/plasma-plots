"""Trapped particles in a tokamak

The fraction of trapped particles in a large-aspect-ratio tokamak, and how it grows towards the plasma edge.

On a flux surface of inverse aspect ratio $\\epsilon = r/R$ the magnetic field varies as
$B \\propto 1/(1 + \\epsilon\\cos\\theta)$, and particles with small enough parallel velocity bounce between
the high-field regions. The trapped fraction is about $1.46\\sqrt{\\epsilon}$ for small $\\epsilon$.
"""

import matplotlib.pyplot as plt
import numpy as np

from plasma_plots.theory import orbits

epsilon = np.linspace(0.01, 0.6, 100)

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.plot(epsilon, orbits.trapped_fraction(epsilon), lw=2, label="exact")
ax.plot(epsilon, 1.46 * np.sqrt(epsilon), "--", lw=1.5, label=r"$1.46\sqrt{\epsilon}$")
ax.set(xlabel=r"inverse aspect ratio $\epsilon$", ylabel="trapped fraction", title="Trapped fraction")
ax.legend(frameon=False)

print(f"epsilon = 0.3: f_t = {orbits.trapped_fraction(0.3):.3f}")
plt.show()
