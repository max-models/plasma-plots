"""Hull cutoff and Brillouin flow in a magnetron

Critical magnetic field of a planar and a coaxial magnetron against anode voltage, and the Brillouin density limit of the electron cloud.

Electrons leaving the cathode just graze the anode when $B = B_c(V) = \\sqrt{2 m V/e}/G$, with $G = d$
for a plane and $G = (r_a^2 - r_c^2)/2r_a$ for coaxial cylinders. Above the cutoff the
discharge is insulated and the electrons form a rotating cloud, whose density cannot exceed the Brillouin value
$n_B = \\epsilon_0 B^2/2m_e$.
"""

import matplotlib.pyplot as plt
import numpy as np

from plasma_plots.theory import magnetron

voltage = np.geomspace(50, 2000, 100)
b_planar = magnetron.hull_cutoff(voltage=voltage, gap=0.05)
b_coax = magnetron.hull_cutoff(voltage=voltage, anode_radius=0.06, cathode_radius=0.01)

field = np.geomspace(1e-3, 0.2, 100)
flow = magnetron.brillouin_flow(field, radius=0.02)

fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
ax_h.loglog(voltage, b_planar * 1e3, lw=2, label="planar, $d$ = 5 cm")
ax_h.loglog(voltage, b_coax * 1e3, lw=2, label=r"coaxial, $r_a$ = 6 cm, $r_c$ = 1 cm")
ax_h.set(
    xlabel="anode voltage (V)",
    ylabel="Hull cutoff field (mT)",
    title="Insulated above each curve",
    ylim=(0.3, 20),
)
ax_h.legend(frameon=False, loc="lower right")

ax_b.loglog(field * 1e3, flow["density"], lw=2, color="C2")
ax_b.set(
    xlabel="magnetic field (mT)",
    ylabel=r"Brillouin density (m$^{-3}$)",
    title="Brillouin flow limit",
)

print(
    f"Brillouin flow at 50 mT: n = {magnetron.brillouin_flow(0.05)['density']:.3e} m^-3"
)
print(
    f"300 V, 5 cm gap: B_c = {magnetron.hull_cutoff(voltage=300.0, gap=0.05) * 1e3:.3f} mT"
)
plt.show()
