"""Sod shock tube

The exact solution of the Riemann problem of gas dynamics, to check a fluid code against.

The classic Sod problem ($\\gamma = 1.4$, density 1 and pressure 1 on the left, 0.125 and 0.1 on the right)
develops a rarefaction fan, a contact discontinuity and a shock. `plasma_plots.theory.exact` gives all of
them in closed form, so a simulation's error can be measured instead of judged by eye.
"""

import matplotlib.pyplot as plt
import numpy as np

from plasma_plots.theory import exact

x = np.linspace(0.0, 1.0, 1000)

fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, layout="constrained")
for t, color in zip((0.05, 0.1, 0.2), ("C0", "C1", "C2")):
    sol = exact.sod_shock_tube(x, t)
    for ax, values in zip(
        axes.flat, (sol.density, sol.velocity, sol.pressure, sol.internal_energy)
    ):
        ax.plot(x, values, color=color, lw=1.8, label=f"t = {t:g}")
for ax, label in zip(axes.flat, ("density", "velocity", "pressure", "internal energy")):
    ax.set_ylabel(label)
for ax in axes[1]:
    ax.set_xlabel("x")
axes[0, 0].legend(frameon=False)

plt.show()
