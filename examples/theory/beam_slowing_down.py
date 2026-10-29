"""Fast-ion slowing down

The slowing down of a fast ion in a plasma: energy loss of a fast deuteron on the electrons and ions of a deuterium plasma, with the critical energy where the two are equal.

Above the critical energy $E_c$ the ion heats mainly the electrons; below it, the ions. The energy
evolves as $dE/dt = -(2E/\\tau_s)[1 + (E_c/E)^{3/2}]$ with the Spitzer time $\\tau_s$, and the ion
thermalises in a finite time. The left panel shows the loss rates against energy, the right the energy against time.
"""

import matplotlib.pyplot as plt
import numpy as np

from plasma_plots.theory import collisions

density = 1e19  # m^-3
temperature = 1000.0  # eV
kw = dict(beam_mass_number=2.0, ion_mass_number=2.0, ln_lambda=17.0)

e_c = collisions.critical_energy(temperature, beam_mass_number=2.0, ion_mass_number=2.0)
tau_s = collisions.slowing_down_time(
    density, temperature, **{"beam_mass_number": 2.0, "ln_lambda": 17.0}
)

energy = np.geomspace(1e3, 2e5, 200)
loss = collisions.beam_stopping_fast_ion(energy, density, temperature, **kw)

t_stop = collisions.slowing_down_time(
    density, temperature, 2.0, ln_lambda=17.0, energy=80e3, critical_energy_ev=e_c
)
t = np.linspace(0, t_stop, 200)
e_t = collisions.beam_stopping_fast_ion(80e3, density, temperature, time=t, **kw)

fig, (ax_l, ax_t) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
ax_l.loglog(energy / 1e3, loss["electrons"], lw=2, label="electrons")
ax_l.loglog(energy / 1e3, loss["ions"], lw=2, label="ions")
ax_l.loglog(energy / 1e3, loss["total"], "k--", lw=1.5, label="total")
ax_l.axvline(e_c / 1e3, color="gray", ls=":", lw=1.5)
ax_l.text(e_c / 1e3 * 1.1, ax_l.get_ylim()[0] * 3, "$E_c$", color="gray")
ax_l.set(
    xlabel="ion energy (keV)", ylabel="energy loss rate (eV/s)", title="Loss rates"
)
ax_l.legend(frameon=False)

ax_t.plot(t * 1e3, e_t / 1e3, lw=2)
ax_t.axhline(
    e_c / 1e3, color="gray", ls=":", lw=1.5, label=f"$E_c$ = {e_c / 1e3:.1f} keV"
)
ax_t.set(xlabel="time (ms)", ylabel="ion energy (keV)", title="80 keV deuteron")
ax_t.legend(frameon=False)

print(
    f"E_c = {e_c / 1e3:.1f} keV, tau_s = {tau_s * 1e3:.0f} ms, 80 keV thermalises in {t_stop * 1e3:.0f} ms"
)
plt.show()
