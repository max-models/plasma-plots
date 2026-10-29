"""KV beam envelope

The KV beam envelope with and without space charge: rms radius of a proton beam in a drift, from the analytic emittance-only hyperbola and from the envelope equation with space charge.

Without space charge the radius follows $x^2 = x_0^2 + (\\epsilon z / x_0)^2$. With the generalized
perveance $K$ the envelope obeys $x'' = \\epsilon^2/x^3 + K/(4x)$ and the beam is blown up much faster
when $K$ is large compared with the emittance term. The perveance of a 30 keV proton beam is computed from its current.
"""

import matplotlib.pyplot as plt
import numpy as np

from plasma_plots.theory import beams
from plasma_plots.theory.parameters import proton_mass

voltage = 30e3  # V
x0 = 5e-3  # m, rms radius of the waist
emittance = 2e-5  # m rad
z = np.linspace(0, 3, 61)

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
ax.plot(
    z,
    beams.kv_envelope(z, x0, 0.0, emittance) * 1e3,
    "k--",
    lw=2,
    label="no space charge (analytic)",
)
for current in (0.2, 0.5, 1.0):
    k_perv = beams.perveance(
        current * 1e-3, voltage, mass=proton_mass, kind="generalized"
    )
    x, _ = beams.kv_envelope_ode(z, x0, 0.0, emittance, perveance=k_perv)
    ax.plot(z, x * 1e3, lw=2, label=f"{current:g} mA, $K$ = {k_perv:.1e}")

x_ode, _ = beams.kv_envelope_ode(z, x0, 0.0, emittance)
ax.plot(z, x_ode * 1e3, ":", color="C3", lw=2, label="no space charge (RK4)")
ax.set(
    xlabel="$z$ (m)", ylabel="rms radius (mm)", title="30 keV proton beam in a drift"
)
ax.legend(frameon=False)

k1 = beams.perveance(1e-3, voltage, mass=proton_mass, kind="generalized")
x1, _ = beams.kv_envelope_ode(z, x0, 0.0, emittance, perveance=k1)
print(
    f"z = 3 m: {beams.kv_envelope(3.0, x0, 0.0, emittance) * 1e3:.2f} mm without, {x1[-1] * 1e3:.2f} mm with 1 mA"
)
plt.show()
